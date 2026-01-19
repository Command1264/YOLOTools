import os
import time
import re
import unicodedata
import threading
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk, ImageOps


def _get_lanczos():
    # Pillow 10+: Image.Resampling.LANCZOS
    # Older: Image.LANCZOS
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


LANCZOS = _get_lanczos()


def imwrite_unicode(path: str, frame_bgr, ext: str = ".jpg") -> bool:
    """
    用 cv2.imencode + Python open 寫檔，支援含中文/Unicode 的路徑與檔名（特別是 Windows）。
    """
    try:
        ok, buf = cv2.imencode(ext, frame_bgr)
        if not ok:
            return False
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False

def sanitize_filename_part(text: str, *, fallback: str = "LINE") -> str:
    r"""
    產生可用於檔名的一段字串：
    - 允許繁中與一般 Unicode
    - 移除 Windows 不允許字元: \ / : * ? " < > |
    - 去掉控制字元
    - 去掉結尾的空白與句點（Windows 不允許）
    """
    if text is None: return fallback

    text = unicodedata.normalize("NFKC", str(text)) # 正規化（避免某些輸入法組合字造成怪問題）
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")  # 移除控制字元
    text = re.sub(r'[\\/:*?"<>|]', "_", text)  # 替換 Windows 不允許字元
    text = text.strip()
    text = text.rstrip(" .")  # Windows 不允許結尾空白或句點

    return text if text else fallback


class VideoFrameExtractorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Frame Extractor (OpenCV)")
        self.root.geometry("920x620")
        self.root.resizable(True, True)  # A方案：可縮放

        # Worker state
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.is_running = False
        self.ended_naturally = False  # 完成 vs 手動停止

        # GUI vars
        self.video_path_var = tk.StringVar(value="")
        self.output_dir_var = tk.StringVar(value=os.path.abspath("./output_frames"))
        self.interval_var = tk.StringVar(value="1.0")    # seconds, float; 0 = every frame
        self.line_id_var = tk.StringVar(value="LINE01")  # 流水線編號

        self.status_var = tk.StringVar(value="狀態：未開始")
        self.saved_count_var = tk.StringVar(value="已擷取：0 張")
        self.progress_var = tk.DoubleVar(value=0.0)

        # Progress tracking
        self.total_frames = 0
        self.current_frame_idx = 0
        self.saved_count = 0

        # Preview state
        self.preview_image_tk = None
        self.overlay_until_ts = 0.0

        # Canvas current size
        self.canvas_w = 640
        self.canvas_h = 360

        # Thread-safe shared frame (worker -> UI)
        self._latest_frame_bgr = None
        self._latest_frame_lock = threading.Lock()
        self._latest_saved_flag = False
        self._last_frame_for_redraw = None  # 用於 resize 時重繪

        self._build_ui()
        # 挪到按下開始時再建立
        # os.makedirs(self.output_dir_var.get(), exist_ok=True)

        # 監聽流水線編號變更：離開欄位或輸入改變都檢查
        self._line_id_editing = False
        self._line_id_was_sanitized = False
        self._line_id_before_sanitize = ""
        self.line_id_var.trace_add("write", self._on_line_id_changed_silent)



    # UI refresh loop
        self._ui_tick()

    # ---------------- UI ----------------

    def _build_ui(self):
        pad_x = 10
        pad_y = 6

        # Top controls
        top = tk.Frame(self.root)
        top.pack(side="top", fill="x", padx=10, pady=10)

        tk.Label(top, text="影片檔：").grid(row=0, column=0, sticky="e", padx=pad_x, pady=pad_y)
        tk.Entry(top, textvariable=self.video_path_var, width=80).grid(row=0, column=1, sticky="we", pady=pad_y)
        tk.Button(top, text="選擇...", command=self.pick_video).grid(row=0, column=2, padx=pad_x, pady=pad_y)

        tk.Label(top, text="輸出資料夾：").grid(row=1, column=0, sticky="e", padx=pad_x, pady=pad_y)
        tk.Entry(top, textvariable=self.output_dir_var, width=80).grid(row=1, column=1, sticky="we", pady=pad_y)
        tk.Button(top, text="選擇...", command=self.pick_output_dir).grid(row=1, column=2, padx=pad_x, pady=pad_y)

        tk.Label(top, text="每隔 n 秒擷取：").grid(row=2, column=0, sticky="e", padx=pad_x, pady=pad_y)
        tk.Entry(top, textvariable=self.interval_var, width=10).grid(row=2, column=1, sticky="w", pady=pad_y)
        tk.Label(top, text="(可小於 1；0 代表每一幀)").grid(row=2, column=1, sticky="w", padx=(90, 0), pady=pad_y)

        tk.Label(top, text="流水線編號：").grid(row=3, column=0, sticky="e", padx=pad_x, pady=pad_y)
        self.line_id_entry = tk.Entry(top, textvariable=self.line_id_var, width=20)
        self.line_id_entry.grid(row=3, column=1, sticky="w", pady=pad_y)
        # self.line_id_entry.bind("<FocusOut>", self._on_line_id_focus_out)


        self.toggle_btn = tk.Button(top, text="開始", width=15, command=self.toggle_start_stop)
        self.toggle_btn.grid(row=3, column=2, padx=pad_x, pady=pad_y)

        tk.Label(top, textvariable=self.status_var, fg="blue").grid(row=4, column=1, sticky="w", pady=(6, 0))
        tk.Label(top, textvariable=self.saved_count_var).grid(row=5, column=1, sticky="w", pady=(2, 0))

        top.grid_columnconfigure(1, weight=1)

        # Progress bar
        pb_row = tk.Frame(self.root)
        pb_row.pack(side="top", fill="x", padx=20, pady=(0, 8))
        tk.Label(pb_row, text="進度：").pack(side="left")

        self.progress_bar = ttk.Progressbar(
            pb_row, orient="horizontal", mode="determinate",
            length=700, variable=self.progress_var, maximum=100.0
        )
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(6, 0))

        # Preview container (A方案：fill+expand)
        self.preview_frame = tk.LabelFrame(self.root, text="預覽")
        self.preview_frame.pack(side="top", padx=10, pady=8, fill="both", expand=True)

        self.preview_canvas = tk.Canvas(self.preview_frame, bg="black", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True, padx=10, pady=10)

        # 監聽 canvas resize
        self.preview_canvas.bind("<Configure>", self._on_preview_resize)

    # ---------------- File dialogs (start from entry path) ----------------

    def pick_video(self):
        current_path = self.video_path_var.get().strip()
        if current_path and os.path.exists(current_path):
            initial_dir = os.path.dirname(current_path) if os.path.isfile(current_path) else current_path
        else:
            initial_dir = os.getcwd()

        path = filedialog.askopenfilename(
            title="選擇影片檔",
            initialdir=initial_dir,
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv *.m4v *.wmv"),
                ("All Files", "*.*")
            ]
        )
        if path:
            self.video_path_var.set(path)

    def pick_output_dir(self):
        current_path = self.output_dir_var.get().strip()
        initial_dir = current_path if (current_path and os.path.exists(current_path)) else os.getcwd()

        path = filedialog.askdirectory(title="選擇輸出資料夾", initialdir=initial_dir)
        if path:
            self.output_dir_var.set(path)
            os.makedirs(path, exist_ok=True)

    # ---------------- Resize handling (A方案核心) ----------------

    def _on_preview_resize(self, event):
        # 更新 canvas 尺寸
        self.canvas_w = max(1, int(event.width))
        self.canvas_h = max(1, int(event.height))

        # 立即用最近一張 frame 重繪（避免你看到「像被裁」的殘影）
        frame = None
        with self._latest_frame_lock:
            if self._last_frame_for_redraw is not None:
                frame = self._last_frame_for_redraw.copy()

        if frame is not None:
            self._render_preview(frame, saved_flag=False)

    # ---------------- Start/Stop ----------------

    def toggle_start_stop(self):
        if not self.is_running:
            self.start()
        else:
            self.stop()

    def start(self):
        video_path = self.video_path_var.get().strip()
        if not video_path or not os.path.isfile(video_path):
            messagebox.showerror("錯誤", "請先選擇有效的影片檔。")
            return

        out_dir = self.output_dir_var.get().strip()
        if not out_dir:
            messagebox.showerror("錯誤", "請設定輸出資料夾。")
            return
        os.makedirs(out_dir, exist_ok=True)

        try:
            interval = float(self.interval_var.get().strip())
            if interval < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("錯誤", "n 秒必須是 >= 0 的數字（可小於 1）。")
            return

        if not self._validate_line_id_on_start():
            return

        line_id = self.line_id_var.get().strip()

        # 挪到這裡，避免一直創建資料夾，同時防止不存在的路徑
        os.makedirs(self.output_dir_var.get(), exist_ok=True)

        # Reset
        self.saved_count = 0
        self.current_frame_idx = 0
        self.total_frames = 0
        self.ended_naturally = False
        self.saved_count_var.set("已擷取：0 張")
        self.progress_var.set(0.0)

        with self._latest_frame_lock:
            self._latest_frame_bgr = None
            self._last_frame_for_redraw = None
            self._latest_saved_flag = False

        self.stop_event.clear()
        self.is_running = True
        self.toggle_btn.config(text="停止")
        self.status_var.set("狀態：擷取中...")

        self.worker_thread = threading.Thread(
            target=self._worker_extract,
            args=(video_path, out_dir, interval, line_id),
            daemon=True
        )
        self.worker_thread.start()

    def stop(self):
        self.stop_event.set()
        self.status_var.set("狀態：停止中...")
        self.toggle_btn.config(state="disabled")
        self.root.after(100, self._poll_worker_stopped)

    def _poll_worker_stopped(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.root.after(100, self._poll_worker_stopped)
            return

        self.is_running = False
        self.toggle_btn.config(state="normal", text="開始")

        if self.ended_naturally and not self.stop_event.is_set():
            self.status_var.set("狀態：已完成")
        elif self.stop_event.is_set():
            self.status_var.set("狀態：已停止")
        else:
            if self.status_var.get() in ("狀態：擷取中...", "狀態：停止中..."):
                self.status_var.set("狀態：已完成")

    # ---------------- UI Tick (preview + progress) ----------------

    def _ui_tick(self):
        frame = None
        saved_flag = False

        with self._latest_frame_lock:
            if self._latest_frame_bgr is not None:
                frame = self._latest_frame_bgr.copy()
                self._latest_frame_bgr = None
            saved_flag = self._latest_saved_flag
            self._latest_saved_flag = False

        if frame is not None:
            # 存一份做 resize 重繪用
            with self._latest_frame_lock:
                self._last_frame_for_redraw = frame.copy()

            self._render_preview(frame, saved_flag=saved_flag)

        # progress update
        if self.is_running:
            if self.total_frames and self.total_frames > 0:
                pct = (self.current_frame_idx / self.total_frames) * 100.0
                pct = max(0.0, min(100.0, pct))
                if self.progress_bar["mode"] != "determinate":
                    self.progress_bar.stop()
                    self.progress_bar.config(mode="determinate")
                self.progress_var.set(pct)
            else:
                if self.progress_bar["mode"] != "indeterminate":
                    self.progress_bar.config(mode="indeterminate")
                    self.progress_bar.start(10)
        else:
            if self.progress_bar["mode"] == "indeterminate":
                self.progress_bar.stop()
                self.progress_bar.config(mode="determinate")

        self.root.after(30, self._ui_tick)

    # ---------------- Preview render (A方案：canvas變多大都OK) ----------------

    def _render_preview(self, frame_bgr, saved_flag: bool):
        """
        1) Canvas 可能不是 16:9，但「顯示框」會在 Canvas 內維持 16:9
        2) 影片影像使用 contain(letterbox)，保證完整顯示、不裁切
        """
        # 先做整個 canvas 的黑底背景
        canvas_w = max(1, self.canvas_w)
        canvas_h = max(1, self.canvas_h)

        # 在 canvas 內算一個 16:9 的 box（最大化塞入）
        box_w = canvas_w
        box_h = int(round(box_w * 9 / 16))
        if box_h > canvas_h:
            box_h = canvas_h
            box_w = int(round(box_h * 16 / 9))

        box_w = max(1, box_w)
        box_h = max(1, box_h)

        # box 左上角（置中）
        box_x = (canvas_w - box_w) // 2
        box_y = (canvas_h - box_h) // 2

        # OpenCV BGR -> RGB PIL
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)

        # contain：保證不裁切
        contained = ImageOps.contain(pil, (box_w, box_h), method=LANCZOS)

        # 先做全 canvas 黑底，再把 contained 貼到 16:9 box 的中心（box 內也黑邊）
        bg = Image.new("RGB", (canvas_w, canvas_h), (0, 0, 0))

        # box 內貼圖位置（box 內置中）
        inner_x = box_x + (box_w - contained.size[0]) // 2
        inner_y = box_y + (box_h - contained.size[1]) // 2
        bg.paste(contained, (inner_x, inner_y))

        now = time.time()
        if saved_flag:
            self.overlay_until_ts = now + 0.35

        # 轉 Tk Image 並畫上 canvas
        self.preview_image_tk = ImageTk.PhotoImage(bg)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_image_tk)

        if now < self.overlay_until_ts:
            self.preview_canvas.create_rectangle(10, 10, 120, 50, fill="black", outline="")
            self.preview_canvas.create_text(65, 30, text="Saved!", fill="white", font=("Arial", 16, "bold"))

    # ---------------- Worker ----------------

    def _worker_extract(self, video_path: str, out_dir: str, interval_sec: float, line_id: str):
        safe_line_id = sanitize_filename_part(line_id, fallback="LINE01")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("錯誤", "無法開啟影片。"))
            self.root.after(0, lambda: self.status_var.set("狀態：錯誤（無法開啟影片）"))
            self.root.after(0, self._poll_worker_stopped)
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        if interval_sec == 0:
            frame_step = 1
        else:
            frame_step = max(1, int(round(interval_sec * fps)))

        frame_idx = 0
        saved_idx = 0
        ended_naturally = False

        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                ended_naturally = True
                break

            with self._latest_frame_lock:
                self._latest_frame_bgr = frame

            saved_flag = False
            if frame_idx % frame_step == 0:
                now = datetime.now()
                ts = now.strftime("%Y%m%d_%H%M%S") + f"_{now.microsecond // 1000:03d}"
                saved_idx += 1
                filename = f"{safe_line_id}_{saved_idx:06d}_{ts}.jpg"
                out_path = os.path.join(out_dir, filename)

                ok = cv2.imwrite(out_path, frame)
                if ok:
                    self.saved_count += 1
                    saved_flag = True
                    self.root.after(0, lambda c=self.saved_count: self.saved_count_var.set(f"已擷取：{c} 張"))

            if saved_flag:
                with self._latest_frame_lock:
                    self._latest_saved_flag = True

            frame_idx += 1
            self.current_frame_idx = frame_idx

            time.sleep(0.001)

        cap.release()

        self.ended_naturally = ended_naturally and (not self.stop_event.is_set())
        self.root.after(0, self._poll_worker_stopped)

    # ---------------- Check String ----------------

    def _on_line_id_changed_silent(self, *args):
        """
        輸入/貼上時：只做靜默自動修正，不跳警告（避免 paste 也一直跳）。
        真正提醒放在 FocusOut 或 Start。
        """
        if self._line_id_editing:
            return

        raw = self.line_id_var.get()
        safe = sanitize_filename_part(raw, fallback="LINE01")

        if raw != safe:
            # 記錄第一次被修正前的內容（用來 FocusOut/Start 時提示）
            if not self._line_id_was_sanitized:
                self._line_id_before_sanitize = raw
            self._line_id_was_sanitized = True

            # 靜默修正
            self._line_id_editing = True
            self.line_id_var.set(safe)
            self._line_id_editing = False

    def _on_line_id_focus_out(self, event=None):
        """
        離開輸入框時：如果曾被自動修正，提示一次。
        """
        if self._line_id_was_sanitized:
            safe = self.line_id_var.get()
            before = self._line_id_before_sanitize
            # 重置旗標，避免一直提示
            self._line_id_was_sanitized = False
            self._line_id_before_sanitize = ""

            messagebox.showwarning(
                "流水線編號已自動修正",
                "流水線編號將用於輸出檔名。\n"
                "你輸入的內容包含檔名不允許字元（例如 \\ / : * ? \" < > |）或尾端空白/句點，已自動修正。\n\n"
                f"原始：{before}\n"
                f"修正：{safe}"
            )

    def _validate_line_id_on_start(self) -> bool:
        """
        開始前再檢查一次；若內容為空或曾被修正但使用者還沒離開欄位，也在這裡提示一次。
        """
        raw = self.line_id_var.get()
        if not raw.strip():
            messagebox.showerror("錯誤", "請填寫流水線編號。")
            return False

        safe = sanitize_filename_part(raw, fallback="LINE01")
        if raw != safe:
            self.line_id_var.set(safe)
            messagebox.showwarning(
                "流水線編號已自動修正",
                "流水線編號將用於輸出檔名。\n"
                "你輸入的內容包含檔名不允許字元或尾端空白/句點，已自動修正。\n\n"
                f"修正後：{safe}"
            )
            return True

        # 若使用者還沒 FocusOut，但 trace 曾修正過，也在 start 前提示一次
        if self._line_id_was_sanitized:
            self._on_line_id_focus_out()
        return True


    def _on_line_id_changed(self, *args):
        """
        使用者輸入時也檢查，但避免每個字都跳警告，做個簡單節流：
        - 只要偵測到非法字元，先把字元替換成底線，但警告最多每 1 秒一次
        """
        if self._line_id_editing:
            return

        raw = self.line_id_var.get()
        safe = sanitize_filename_part(raw, fallback="LINE01")

        # 若 raw -> safe 有變化，代表包含非法字元/尾端問題/控制字元
        if raw != safe:
            now = time.time()
            show_warn = now >= getattr(self, "_line_id_warn_cooldown_until", 0.0)
            # 自動修正欄位內容
            self._line_id_editing = True
            self.line_id_var.set(safe)
            self._line_id_editing = False

            if show_warn:
                self._line_id_warn_cooldown_until = now + 1.0
                messagebox.showwarning(
                    "流水線編號已自動修正",
                    "你輸入的流水線編號包含檔名不允許的字元（例如 \\ / : * ? \" < > |）\n"
                    "或尾端空白/句點，已自動替換成安全字串。\n\n"
                    f"修正後：{safe}"
                )

    def _validate_line_id_on_start(self) -> bool:
        """
        開始前再做一次嚴格檢查（保險），並提醒使用者。
        回傳 True 表示可繼續；False 表示不可開始。
        """
        raw = self.line_id_var.get().strip()
        if not raw:
            messagebox.showerror("錯誤", "請填寫流水線編號。")
            return False

        safe = sanitize_filename_part(raw, fallback="LINE01")
        if raw != safe:
            # 自動修正並提醒（開始前提示更明確）
            self.line_id_var.set(safe)
            messagebox.showwarning(
                "流水線編號已自動修正",
                "流水線編號將用於輸出檔名。\n"
                "你輸入的內容含有檔名不允許字元或格式問題，已自動修正。\n\n"
                f"修正後：{safe}"
            )
        return True

    def _get_safe_line_id(self) -> str:
        """
        統一取得 worker 要用的安全流水線編號（避免任何漏網）。
        """
        return sanitize_filename_part(self.line_id_var.get(), fallback="LINE01")



def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass

    VideoFrameExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
