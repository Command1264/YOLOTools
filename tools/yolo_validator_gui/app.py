import json
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

import sys
from pathlib import Path as _Path

APP_DIR = _Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
SERVER_DIR = ROOT_DIR / "yolo_server_gui"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from yolo_engine import YoloEngine


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLOv26 模型驗證器")
        self.geometry("1100x720")
        self.minsize(980, 640)

        self.var_model = tk.StringVar()
        self.var_input = tk.StringVar()
        self.var_conf = tk.DoubleVar(value=0.70)
        self.var_iou = tk.DoubleVar(value=0.45)
        self.var_conf_str = tk.StringVar(value="0.70")
        self.var_iou_str = tk.StringVar(value="0.45")
        self.var_device = tk.StringVar(value="")
        self.var_show = tk.BooleanVar(value=True)
        self.var_ignore_conf = tk.BooleanVar(value=False)
        self.var_ignore_iou = tk.BooleanVar(value=False)
        self.var_interval = tk.DoubleVar(value=1.0)
        self.var_order = tk.StringVar(value="圖片優先")
        self._order_map = {
            "圖片優先": "images_first",
            "影片優先": "videos_first",
        }
        self._order_map_rev = {v: k for k, v in self._order_map.items()}
        self._last_model_dir: Optional[Path] = None
        self._last_input_dir: Optional[Path] = None

        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._last_image: Optional[ImageTk.PhotoImage] = None
        self._ui_token = 0
        self._conf_prev: Optional[float] = None
        self._iou_prev: Optional[float] = None
        self._loading_config = False
        self._config_path = Path(__file__).resolve().parent / "yolo_validator_config.json"
        self._syncing_scale = False

        self._build_ui()

    def _build_ui(self):
        pad = 10
        root = ttk.Frame(self)
        root.pack(fill="both", expand=True, padx=pad, pady=pad)

        path_box = ttk.LabelFrame(root, text="模型與檔案")
        path_box.pack(fill="x", padx=2, pady=6)

        ttk.Label(path_box, text="YOLO 模型 (.pt):").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.ent_model = ttk.Entry(path_box, textvariable=self.var_model)
        self.ent_model.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.ent_model.bind("<FocusOut>", lambda _e: self._on_model_entry())
        self.ent_model.bind("<Return>", lambda _e: self._on_model_entry())
        ttk.Button(path_box, text="瀏覽...", command=self.browse_model).grid(row=0, column=2, padx=8, pady=6)

        ttk.Label(path_box, text="圖片或影片:").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.ent_input = ttk.Entry(path_box, textvariable=self.var_input)
        self.ent_input.grid(row=1, column=1, sticky="ew", padx=8, pady=6)
        self.ent_input.bind("<FocusOut>", lambda _e: self._on_input_entry())
        self.ent_input.bind("<FocusOut>", lambda _e: self._on_input_entry())
        self.ent_input.bind("<Return>", lambda _e: self._on_input_entry())
        ttk.Button(path_box, text="瀏覽檔案...", command=self.browse_input).grid(row=1, column=2, padx=8, pady=6)
        ttk.Button(path_box, text="瀏覽資料夾...", command=self.browse_input_dir).grid(row=1, column=3, padx=8, pady=6)

        path_box.columnconfigure(1, weight=1)

        opt_box = ttk.LabelFrame(root, text="推論設定")
        opt_box.pack(fill="x", padx=2, pady=6)

        ttk.Label(opt_box, text="conf:").grid(row=0, column=0, sticky="w", padx=8, pady=6)
        self.scale_conf = ttk.Scale(
            opt_box,
            from_=0.01,
            to=1.0,
            orient="horizontal",
            command=lambda v: self._sync_from_scale("conf", v),
        )
        self.scale_conf.grid(row=0, column=1, sticky="ew", padx=8, pady=6)
        self.ent_conf = ttk.Entry(opt_box, textvariable=self.var_conf_str, width=8)
        self.ent_conf.grid(row=0, column=2, sticky="w", padx=4, pady=6)
        self.ent_conf.bind("<FocusOut>", lambda _e: self._validate_entry("conf"))
        self.ent_conf.bind("<Return>", lambda _e: self._validate_entry("conf"))

        ttk.Label(opt_box, text="iou:").grid(row=0, column=3, sticky="w", padx=8, pady=6)
        self.scale_iou = ttk.Scale(
            opt_box,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            command=lambda v: self._sync_from_scale("iou", v),
        )
        self.scale_iou.grid(row=0, column=4, sticky="ew", padx=8, pady=6)
        self.ent_iou = ttk.Entry(opt_box, textvariable=self.var_iou_str, width=8)
        self.ent_iou.grid(row=0, column=5, sticky="w", padx=4, pady=6)
        self.ent_iou.bind("<FocusOut>", lambda _e: self._validate_entry("iou"))
        self.ent_iou.bind("<Return>", lambda _e: self._validate_entry("iou"))

        ttk.Label(opt_box, text="device (空白=auto):").grid(row=1, column=0, sticky="w", padx=8, pady=6)
        self.ent_device = ttk.Entry(opt_box, textvariable=self.var_device, width=10)
        self.ent_device.grid(row=1, column=1, sticky="w", padx=8, pady=6)
        self.ent_device.bind("<FocusOut>", lambda _e: self._save_config())
        self.ent_device.bind("<Return>", lambda _e: self._save_config())
        ttk.Checkbutton(opt_box, text="顯示 YOLO 判斷框與標籤", variable=self.var_show, command=self._on_toggle_show).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=8, pady=6
        )
        ttk.Checkbutton(opt_box, text="忽略 conf", variable=self.var_ignore_conf, command=self._on_ignore_conf).grid(
            row=1, column=4, sticky="w", padx=8, pady=6
        )
        ttk.Checkbutton(opt_box, text="忽略 iou", variable=self.var_ignore_iou, command=self._on_ignore_iou).grid(
            row=1, column=5, sticky="w", padx=8, pady=6
        )

        ttk.Label(opt_box, text="圖片切換/影片間隔(秒):").grid(row=2, column=0, sticky="w", padx=8, pady=6)
        self.ent_interval = ttk.Entry(opt_box, textvariable=self.var_interval, width=8)
        self.ent_interval.grid(row=2, column=1, sticky="w", padx=8, pady=6)
        self.ent_interval.bind("<FocusOut>", lambda _e: self._on_interval_entry())
        self.ent_interval.bind("<Return>", lambda _e: self._on_interval_entry())

        ttk.Label(opt_box, text="播放順序:").grid(row=2, column=2, sticky="w", padx=8, pady=6)
        self.cmb_order = ttk.Combobox(
            opt_box,
            textvariable=self.var_order,
            state="readonly",
            values=["圖片優先", "影片優先"],
            width=14,
        )
        self.cmb_order.grid(row=2, column=3, sticky="w", padx=8, pady=6)
        self.cmb_order.bind("<<ComboboxSelected>>", lambda _e: self._save_config())

        opt_box.columnconfigure(1, weight=1)
        opt_box.columnconfigure(4, weight=1)

        act = ttk.Frame(root)
        act.pack(fill="x", padx=2, pady=6)
        self.btn_start = ttk.Button(act, text="開始", command=self.start)
        self.btn_start.pack(side="left", padx=6)
        self.btn_stop = ttk.Button(act, text="停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        self.lbl_status = ttk.Label(act, text="就緒")
        self.lbl_status.pack(side="left", padx=10)
        self.lbl_device = ttk.Label(act, text="device: -")
        self.lbl_device.pack(side="left", padx=10)

        paned = tk.PanedWindow(root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=2, pady=6)

        view = ttk.LabelFrame(paned, text="預覽")
        self.canvas = tk.Canvas(view, bg="#111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        log_frame = ttk.LabelFrame(paned, text="Log")
        self.log_box = tk.Text(log_frame, height=8, wrap="word")
        self.log_box.pack(fill="both", expand=True)

        paned.add(view)
        paned.add(log_frame)
        paned.paneconfigure(log_frame, minsize=90)

        self._canvas_img_id = None
        self._last_frame_bgr = None

        self._paned = paned
        self._log_frame = log_frame
        self._log_lines = 5

        self.log("就緒。請選擇模型與圖片/影片。")
        self._load_config()
        self._sync_from_value("conf", self.var_conf.get())
        self._sync_from_value("iou", self.var_iou.get())
        self._apply_ignore_state()
        self.after(0, lambda: self._set_log_height(self._paned, self._log_frame, self._log_lines))
        self.canvas.bind("<Configure>", lambda _e: self._redraw_last_frame())
        self._paned.bind("<Configure>", lambda _e: self._set_log_height(self._paned, self._log_frame, self._log_lines))
        self.after(100, self._try_preview_on_start)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{ts}] {msg}\n")
        self.log_box.see("end")
        self.update_idletasks()

    def browse_model(self):
        initial = self._resolve_initial_dir(self.var_model.get(), self._last_model_dir)
        p = filedialog.askopenfilename(
            title="選擇 YOLO 模型",
            initialdir=initial,
            filetypes=[("YOLO Model", "*.pt"), ("All", "*.*")],
        )
        if p:
            self.var_model.set(p)
            self._last_model_dir = Path(p).parent
            self._stop_if_running()
            self._save_config()

    def browse_input(self):
        initial = self._resolve_initial_dir(self.var_input.get(), self._last_input_dir)
        p = filedialog.askopenfilename(
            title="選擇圖片或影片",
            initialdir=initial,
            filetypes=[
                ("Images/Video", "*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.mp4;*.avi;*.mov;*.mkv;*.wmv"),
                ("All", "*.*"),
            ],
        )
        if p:
            self.var_input.set(p)
            self._last_input_dir = Path(p).parent
            self._preview_input(Path(p))
            self._stop_if_running()
            self._save_config()

    def browse_input_dir(self):
        initial = self._resolve_initial_dir(self.var_input.get(), self._last_input_dir)
        p = filedialog.askdirectory(
            title="選擇資料夾",
            initialdir=initial,
        )
        if p:
            self.var_input.set(p)
            self._last_input_dir = Path(p)
            self._preview_input(Path(p))
            self._stop_if_running()
            self._save_config()

    def stop(self):
        self._stop_event.set()
        self.lbl_status.config(text="停止中...")

    def start(self):
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("執行中", "目前正在執行。")
            return

        model_path = Path(self.var_model.get().strip())
        input_path = Path(self.var_input.get().strip())
        if not model_path.exists():
            messagebox.showerror("錯誤", "模型檔不存在。")
            return
        if not input_path.exists():
            messagebox.showerror("錯誤", "輸入檔不存在。")
            return
        if input_path.is_dir():
            if not self._collect_folder_items(input_path):
                messagebox.showerror("錯誤", "資料夾內找不到可用的圖片或影片。")
                return

        self._validate_entry("conf")
        self._validate_entry("iou")
        self._on_interval_entry()

        self._stop_event.clear()
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="執行中...")

        self._ui_token += 1
        token = self._ui_token
        self._worker = threading.Thread(
            target=self._run,
            args=(model_path, input_path, self.var_device.get(), token),
            daemon=True,
        )
        self._worker.start()

    def _run(self, model_path: Path, input_path: Path, device: str, token: int):
        try:
            engine = YoloEngine(str(model_path), device=device.strip())
            engine.load()
            self._ui_device(engine.device_name)
            if engine.cuda_available and engine.device_name.startswith("cpu"):
                self.log("偵測到 CUDA 可用，但模型仍在 CPU。可嘗試 device=cuda:0。")
            if input_path.is_dir():
                self._run_folder(engine, input_path, token)
            else:
                suffix = input_path.suffix.lower()
                if suffix in IMG_EXTS:
                    self._run_image(engine, input_path, token)
                elif suffix in VID_EXTS:
                    self._run_video(engine, input_path, token)
                else:
                    self._ui_error("不支援的檔案格式。")
        except Exception as e:
            self._ui_error(f"執行失敗: {e}")
        finally:
            self._ui_done()

    def _run_image(self, engine: YoloEngine, path: Path, token: int):
        frame = cv2.imread(str(path))
        if frame is None:
            self._ui_error("讀取圖片失敗。")
            return

        self._process_and_show(engine, frame, token)

    def _run_video(self, engine: YoloEngine, path: Path, token: int):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self._ui_error("讀取影片失敗。")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.03

        while not self._stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            self._process_and_show(engine, frame, token)
            time.sleep(delay)

        cap.release()

    def _run_folder(self, engine: YoloEngine, folder: Path, token: int):
        images, videos = self._collect_folder_items(folder)
        if not images and not videos:
            self._ui_error("資料夾內沒有可用的圖片或影片。")
            return
        order = self._order_map.get(self.var_order.get(), "images_first")
        if order == "videos_first":
            sequence = [("video", p) for p in videos] + [("image", p) for p in images]
        else:
            sequence = [("image", p) for p in images] + [("video", p) for p in videos]

        interval = max(0.0, float(self.var_interval.get()))
        for kind, path in sequence:
            if self._stop_event.is_set() or token != self._ui_token:
                return
            if kind == "image":
                self._run_image(engine, path, token)
                self._sleep_interval(interval, token)
            else:
                self._run_video(engine, path, token)
                self._sleep_interval(interval, token)

    def _collect_folder_items(self, folder: Path):
        images = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])
        videos = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VID_EXTS])
        return images, videos

    def _sleep_interval(self, seconds: float, token: int):
        if seconds <= 0:
            return
        end_time = time.time() + seconds
        while time.time() < end_time:
            if self._stop_event.is_set() or token != self._ui_token:
                return
            time.sleep(0.05)

    def _process_and_show(self, engine: YoloEngine, frame_bgr, token: int):
        if token != self._ui_token:
            return
        if self.var_show.get():
            conf = float(self.var_conf.get())
            iou = float(self.var_iou.get())
            if self.var_ignore_conf.get():
                conf = 0.01
            if self.var_ignore_iou.get():
                iou = 1.0
            result, dets = engine.infer(frame_bgr, conf=conf, iou=iou)
            if result is not None:
                annotated = result.plot()
            else:
                annotated = frame_bgr
            info = self._format_dets(dets)
            self._ui_update(annotated, info, token)
        else:
            self._ui_update(frame_bgr, "顯示原始影像（未顯示判斷）", token)

    def _format_dets(self, dets) -> str:
        if not dets:
            return "未偵測到物件"
        top = dets[:5]
        summary = ", ".join([f"{d.class_name} {d.conf:.2f}" for d in top])
        return f"偵測 {len(dets)} 個: {summary}"

    def _ui_update(self, frame_bgr, info: str, token: Optional[int] = None):
        self._last_frame_bgr = frame_bgr
        def _update():
            if token is not None and token != self._ui_token:
                return
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            img = self._fit_image(img, (self.canvas.winfo_width(), self.canvas.winfo_height()))
            self._last_image = ImageTk.PhotoImage(img)
            w = self.canvas.winfo_width()
            h = self.canvas.winfo_height()
            if w <= 0 or h <= 0:
                w, h = img.width, img.height
            if self._canvas_img_id is None:
                self._canvas_img_id = self.canvas.create_image(w // 2, h // 2, image=self._last_image, anchor="center")
            else:
                self.canvas.itemconfigure(self._canvas_img_id, image=self._last_image)
                self.canvas.coords(self._canvas_img_id, w // 2, h // 2)
            self.lbl_status.config(text=info)

        self.after(0, _update)

    def _redraw_last_frame(self):
        if self._last_frame_bgr is None:
            return
        rgb = cv2.cvtColor(self._last_frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = self._fit_image(img, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        self._last_image = ImageTk.PhotoImage(img)
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 0 or h <= 0:
            w, h = img.width, img.height
        if self._canvas_img_id is None:
            self._canvas_img_id = self.canvas.create_image(w // 2, h // 2, image=self._last_image, anchor="center")
        else:
            self.canvas.itemconfigure(self._canvas_img_id, image=self._last_image)
            self.canvas.coords(self._canvas_img_id, w // 2, h // 2)

    def _ui_device(self, device_name: str):
        self.after(0, lambda: self.lbl_device.config(text=f"device: {device_name}"))

    def _ui_error(self, msg: str):
        self.after(0, lambda: messagebox.showerror("錯誤", msg))
        self.after(0, lambda: self.log(msg))

    def _ui_done(self):
        def _done():
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")
            if not self._stop_event.is_set():
                self.lbl_status.config(text="完成")
            else:
                self.lbl_status.config(text="已停止")

        self.after(0, _done)

    @staticmethod
    def _fit_image(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
        w, h = size
        if w <= 0 or h <= 0:
            return img
        img_ratio = img.width / img.height
        box_ratio = w / h
        if img_ratio > box_ratio:
            new_w = w
            new_h = int(w / img_ratio)
        else:
            new_h = h
            new_w = int(h * img_ratio)
        return img.resize((max(1, new_w), max(1, new_h)), Image.BILINEAR)

    def _stop_if_running(self):
        if self._worker and self._worker.is_alive():
            self._ui_token += 1
            self._stop_event.set()
            self.lbl_status.config(text="已停止（重新選擇）")

    def _preview_input(self, path: Path):
        try:
            if path.is_dir():
                images, videos = self._collect_folder_items(path)
                order = self._order_map.get(self.var_order.get(), "images_first")
                if order == "videos_first":
                    pick = videos[0] if videos else (images[0] if images else None)
                else:
                    pick = images[0] if images else (videos[0] if videos else None)
                if pick is None:
                    return
                self._preview_input(pick)
                return
            suffix = path.suffix.lower()
            if suffix in IMG_EXTS:
                frame = cv2.imread(str(path))
                if frame is None:
                    return
                self._ui_update(frame, "預覽圖片")
                return
            if suffix in VID_EXTS:
                cap = cv2.VideoCapture(str(path))
                if not cap.isOpened():
                    return
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    self._ui_update(frame, "預覽影片第一幀")
        except Exception:
            return

    def _on_model_entry(self):
        raw = (self.var_model.get() or "").strip()
        if raw:
            p = Path(raw)
            if p.exists():
                self._last_model_dir = p if p.is_dir() else p.parent
            else:
                self.log("模型路徑不存在。")
        self._save_config()

    def _on_input_entry(self):
        raw = (self.var_input.get() or "").strip()
        if raw:
            p = Path(raw)
            if p.exists():
                self._last_input_dir = p if p.is_dir() else p.parent
                self._preview_input(p)
            else:
                self.log("圖片或影片路徑不存在。")
        self._save_config()

    def _on_toggle_show(self):
        self._save_config()

    def _on_interval_entry(self):
        try:
            v = float(self.var_interval.get())
        except Exception:
            v = 1.0
        if v < 0:
            v = 0.0
        self.var_interval.set(v)
        self._save_config()

    def _apply_ignore_state(self):
        if self.var_ignore_conf.get():
            if self._conf_prev is None:
                self._conf_prev = float(self.var_conf.get())
            self._set_conf_value(0.01)
            self.ent_conf.configure(state="disabled")
            self.scale_conf.configure(state="disabled")
        else:
            self.ent_conf.configure(state="normal")
            self.scale_conf.configure(state="normal")

        if self.var_ignore_iou.get():
            if self._iou_prev is None:
                self._iou_prev = float(self.var_iou.get())
            self._set_iou_value(1.0)
            self.ent_iou.configure(state="disabled")
            self.scale_iou.configure(state="disabled")
        else:
            self.ent_iou.configure(state="normal")
            self.scale_iou.configure(state="normal")

    def _on_ignore_conf(self):
        if self.var_ignore_conf.get():
            self._conf_prev = float(self.var_conf.get())
            self._set_conf_value(0.01)
            self.ent_conf.configure(state="disabled")
            self.scale_conf.configure(state="disabled")
        else:
            self.ent_conf.configure(state="normal")
            self.scale_conf.configure(state="normal")
            if self._conf_prev is not None:
                self._set_conf_value(self._conf_prev)
                self._conf_prev = None
        self._save_config()

    def _on_ignore_iou(self):
        if self.var_ignore_iou.get():
            self._iou_prev = float(self.var_iou.get())
            self._set_iou_value(1.0)
            self.ent_iou.configure(state="disabled")
            self.scale_iou.configure(state="disabled")
        else:
            self.ent_iou.configure(state="normal")
            self.scale_iou.configure(state="normal")
            if self._iou_prev is not None:
                self._set_iou_value(self._iou_prev)
                self._iou_prev = None
        self._save_config()

    def _set_conf_value(self, v: float, allow_zero: bool = False):
        min_v = 0.0 if allow_zero else 0.01
        v = max(min_v, min(1.0, float(v)))
        self.var_conf.set(v)
        self.var_conf_str.set(f"{v:.2f}")
        if not self._syncing_scale:
            if v <= 0.0:
                self._syncing_scale = True
                try:
                    self.scale_conf.set(0.01)
                finally:
                    self._syncing_scale = False
                return
            self.scale_conf.set(v)

    def _set_iou_value(self, v: float):
        v = max(0.0, min(1.0, float(v)))
        self.var_iou.set(v)
        self.var_iou_str.set(f"{v:.2f}")
        if not self._syncing_scale:
            self.scale_iou.set(v)

    def _load_config(self):
        if not self._config_path.exists():
            return
        try:
            self._loading_config = True
            data = json.loads(self._config_path.read_text(encoding="utf-8"))
            self.var_model.set(data.get("model_path", self.var_model.get()))
            self.var_input.set(data.get("input_path", self.var_input.get()))
            conf = float(data.get("conf", self.var_conf.get()))
            iou = float(data.get("iou", self.var_iou.get()))
            conf = max(0.0, min(1.0, conf))
            iou = max(0.0, min(1.0, iou))
            self.var_conf.set(conf)
            self.var_iou.set(iou)
            self.var_conf_str.set(f"{conf:.2f}")
            self.var_iou_str.set(f"{iou:.2f}")
            self.var_device.set(data.get("device", self.var_device.get()))
            self.var_show.set(bool(data.get("show", True)))
            self.var_ignore_conf.set(bool(data.get("ignore_conf", False)))
            self.var_ignore_iou.set(bool(data.get("ignore_iou", False)))
            interval = float(data.get("interval_sec", self.var_interval.get()))
            if interval < 0:
                interval = 0.0
            self.var_interval.set(interval)
            order_raw = data.get("order", self._order_map.get(self.var_order.get(), "images_first"))
            self.var_order.set(self._order_map_rev.get(order_raw, "圖片優先"))

            if self.var_ignore_conf.get():
                self._conf_prev = conf
            if self.var_ignore_iou.get():
                self._iou_prev = iou

            if self.var_model.get().strip():
                p = Path(self.var_model.get().strip())
                if p.exists():
                    self._last_model_dir = p if p.is_dir() else p.parent
            if self.var_input.get().strip():
                p = Path(self.var_input.get().strip())
                if p.exists():
                    self._last_input_dir = p if p.is_dir() else p.parent
        except Exception:
            return
        finally:
            self._loading_config = False

    def _save_config(self):
        if self._loading_config:
            return
        try:
            conf_val = float(self.var_conf.get())
            iou_val = float(self.var_iou.get())
            if self.var_ignore_conf.get() and self._conf_prev is not None:
                conf_val = float(self._conf_prev)
            if self.var_ignore_iou.get() and self._iou_prev is not None:
                iou_val = float(self._iou_prev)
            data = {
                "model_path": self.var_model.get(),
                "input_path": self.var_input.get(),
                "conf": conf_val,
                "iou": iou_val,
                "device": self.var_device.get(),
                "show": bool(self.var_show.get()),
                "ignore_conf": bool(self.var_ignore_conf.get()),
                "ignore_iou": bool(self.var_ignore_iou.get()),
                "interval_sec": float(self.var_interval.get()),
                "order": self._order_map.get(self.var_order.get(), "images_first"),
            }
            self._config_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")
        except Exception:
            return

    @staticmethod
    def _resolve_initial_dir(path_value: str, last_dir: Optional[Path]) -> str:
        try:
            raw = (path_value or "").strip()
            if raw:
                p = Path(raw)
                if p.exists():
                    return str(p if p.is_dir() else p.parent)
            if last_dir and last_dir.exists():
                return str(last_dir)
        except Exception:
            pass
        return str(Path.cwd())

    def _set_log_height(self, paned: tk.PanedWindow, log_frame: ttk.LabelFrame, lines: int):
        try:
            self.update_idletasks()
            line_px = int(self.log_box["font"].split()[-1]) + 6
        except Exception:
            line_px = 18
        log_h = max(90, lines * line_px + 18)
        total = paned.winfo_height()
        if total <= 0:
            return
        sash = max(50, total - log_h)
        try:
            paned.sash_place(0, 0, sash)
        except Exception:
            pass

    def _sync_from_scale(self, which: str, value: str):
        if which == "conf" and self.var_ignore_conf.get():
            return
        if which == "iou" and self.var_ignore_iou.get():
            return
        if self._syncing_scale:
            return
        try:
            v = float(value)
        except Exception:
            return
        if which == "conf":
            v = max(0.01, min(1.0, v))
        else:
            v = max(0.0, min(1.0, v))
        self._syncing_scale = True
        try:
            if which == "conf":
                self.var_conf.set(v)
                self.var_conf_str.set(f"{v:.2f}")
            else:
                self.var_iou.set(v)
                self.var_iou_str.set(f"{v:.2f}")
        finally:
            self._syncing_scale = False
        self._save_config()

    def _sync_from_value(self, which: str, value: float):
        if which == "conf":
            raw = float(value)
            if raw <= 0.0:
                self.var_conf_str.set("0.00")
                self._syncing_scale = True
                try:
                    self.scale_conf.set(0.01)
                finally:
                    self._syncing_scale = False
                return
            v = max(0.01, min(1.0, raw))
            self.var_conf_str.set(f"{v:.2f}")
            self.scale_conf.set(v)
        else:
            v = max(0.0, min(1.0, float(value)))
            self.var_iou_str.set(f"{v:.2f}")
            self.scale_iou.set(v)

    def _validate_entry(self, which: str):
        if which == "conf" and self.var_ignore_conf.get():
            return
        if which == "iou" and self.var_ignore_iou.get():
            return
        raw = self.var_conf_str.get() if which == "conf" else self.var_iou_str.get()
        try:
            v = float(raw)
        except Exception:
            v = 0.70 if which == "conf" else 0.45
        if v > 1:
            try:
                digits = len(str(int(v)))
                v = v / (10 ** digits)
            except Exception:
                v = 1.0
        v = max(0.0, min(1.0, v))
        if which == "conf":
            if v == 0.0:
                self._set_conf_value(0.0, allow_zero=True)
            else:
                self._set_conf_value(v)
        else:
            self._set_iou_value(v)
        self._save_config()

    def _try_preview_on_start(self):
        raw = (self.var_input.get() or "").strip()
        if not raw:
            return
        p = Path(raw)
        if p.exists() and p.is_file():
            self._preview_input(p)


if __name__ == "__main__":
    app = App()
    app.mainloop()
