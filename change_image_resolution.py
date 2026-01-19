import os
import threading
import time
import platform
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import cv2
import numpy as np
from PIL import Image, ImageTk


# =========================
# 取得程式所在資料夾（沒 __file__ 時退回 cwd）
# =========================
def get_app_dir() -> str:
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except Exception:
        return os.getcwd()


# =========================
# 對話框 initialdir 規則：
# - 如果目前路徑為空：用程式所在資料夾
# - 如果有路徑：優先開啟該路徑（檔案則開其所在資料夾）
# - 路徑無效：回到程式所在資料夾
# =========================
def initial_dir_from(path_str: str) -> str:
    app_dir = get_app_dir()
    p = (path_str or "").strip()
    if not p:
        return app_dir
    if os.path.isdir(p):
        return p
    if os.path.isfile(p):
        return os.path.dirname(p) or app_dir
    return app_dir


# =========================
# Windows：強制把「原生對話框」搬到主視窗中心（best-effort）
# =========================
def _win_center_dialog_over_root_async(root: tk.Tk, timeout_sec: float = 2.0):
    if platform.system().lower() != "windows":
        return

    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    # typedef BOOL (CALLBACK* WNDENUMPROC)(HWND, LPARAM);
    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    EnumWindows = user32.EnumWindows
    EnumWindows.argtypes = [WNDENUMPROC, wintypes.LPARAM]
    EnumWindows.restype = wintypes.BOOL

    GetClassNameW = user32.GetClassNameW
    GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    GetClassNameW.restype = ctypes.c_int

    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    GetWindowTextW.restype = ctypes.c_int

    IsWindowVisible = user32.IsWindowVisible
    IsWindowVisible.argtypes = [wintypes.HWND]
    IsWindowVisible.restype = wintypes.BOOL

    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    GetWindowThreadProcessId.restype = wintypes.DWORD

    GetWindowRect = user32.GetWindowRect
    GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    GetWindowRect.restype = wintypes.BOOL

    MoveWindow = user32.MoveWindow
    MoveWindow.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.BOOL]
    MoveWindow.restype = wintypes.BOOL

    GetCurrentProcessId = kernel32.GetCurrentProcessId
    GetCurrentProcessId.argtypes = []
    GetCurrentProcessId.restype = wintypes.DWORD

    pid = GetCurrentProcessId()

    # 常見標題（不同語系/版本會不一樣）
    possible_titles = {
        "Color", "Choose Color", "Select Color",
        "色彩", "選擇色彩", "選擇顏色", "選色", "顏色"
    }

    def worker():
        end = time.time() + timeout_sec

        try:
            root.update_idletasks()
            rx = root.winfo_rootx()
            ry = root.winfo_rooty()
            rw = root.winfo_width()
            rh = root.winfo_height()
            rcx = rx + rw // 2
            rcy = ry + rh // 2
        except Exception:
            rcx, rcy = 600, 400

        found_hwnd = None

        @WNDENUMPROC
        def enum_proc(hwnd, lparam):
            nonlocal found_hwnd
            if found_hwnd:
                return False

            if not IsWindowVisible(hwnd):
                return True

            wpid = wintypes.DWORD()
            GetWindowThreadProcessId(hwnd, ctypes.byref(wpid))
            if wpid.value != pid:
                return True

            cls = ctypes.create_unicode_buffer(256)
            GetClassNameW(hwnd, cls, 256)
            if cls.value != "#32770":  # 系統對話框 class
                return True

            title = ctypes.create_unicode_buffer(512)
            GetWindowTextW(hwnd, title, 512)
            t = (title.value or "").strip()

            # 有些版本標題可能空的；空標題也放行
            if t and (t not in possible_titles):
                return True

            found_hwnd = hwnd
            return False

        while time.time() < end and not found_hwnd:
            try:
                EnumWindows(enum_proc, 0)
            except Exception:
                return
            if not found_hwnd:
                time.sleep(0.02)

        if not found_hwnd:
            return

        rect = wintypes.RECT()
        if not GetWindowRect(found_hwnd, ctypes.byref(rect)):
            return
        w = rect.right - rect.left
        h = rect.bottom - rect.top
        if w <= 0 or h <= 0:
            return

        x = int(rcx - w / 2)
        y = int(rcy - h / 2)
        x = max(0, x)
        y = max(0, y)

        try:
            MoveWindow(found_hwnd, x, y, w, h, True)
        except Exception:
            return

    threading.Thread(target=worker, daemon=True).start()


# =========================
# OpenCV 讀寫（支援中文路徑）
# =========================
def cv_imread_unicode(path: str):
    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def cv_imwrite_unicode(path: str, img_bgr, *, ext: str, params=None):
    if not ext.startswith("."):
        ext = "." + ext
    ok, buf = cv2.imencode(ext, img_bgr, params if params else [])
    if not ok:
        return False
    try:
        buf.tofile(path)
        return True
    except Exception:
        return False


def is_image_file(fn: str):
    ext = os.path.splitext(fn)[1].lower()
    return ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]


def hex_to_bgr(hex_str: str):
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError("Hex must be 6 digits")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def bgr_to_hex(bgr):
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


def ensure_unique_path(path: str):
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    i = 1
    while True:
        cand = f"{base}({i}){ext}"
        if not os.path.exists(cand):
            return cand
        i += 1


ANCHORS = [
    "左上", "上中", "右上",
    "左中", "置中", "右中",
    "左下", "下中", "右下",
]


def compute_offset(canvas_w, canvas_h, img_w, img_h, anchor: str):
    if anchor in ("左上", "左中", "左下"):
        x = 0
    elif anchor in ("上中", "置中", "下中"):
        x = (canvas_w - img_w) // 2
    else:
        x = canvas_w - img_w

    if anchor in ("左上", "上中", "右上"):
        y = 0
    elif anchor in ("左中", "置中", "右中"):
        y = (canvas_h - img_h) // 2
    else:
        y = canvas_h - img_h

    return max(0, x), max(0, y)


def resize_letterbox(img_bgr, target_w: int, target_h: int, fill_bgr=(0, 0, 0), anchor="置中"):
    if img_bgr is None:
        return None

    h, w = img_bgr.shape[:2]
    if w <= 0 or h <= 0 or target_w <= 0 or target_h <= 0:
        return None

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        img_bgr, (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    )

    canvas = np.full((target_h, target_w, 3), fill_bgr, dtype=np.uint8)
    x, y = compute_offset(target_w, target_h, new_w, new_h, anchor)
    canvas[y:y + new_h, x:x + new_w] = resized
    return canvas


class ImageResizerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Image Resizer (OpenCV) - 等比例縮放 + 補色 + 批次")
        self.root.geometry("1300x800")
        self.root.minsize(1200, 800)
        root.state('zoomed')   # Windows only

        self.fill_bgr = (0, 0, 0)
        self.worker_thread = None
        self.stop_flag = False

        self.var_input = tk.StringVar()
        self.var_output = tk.StringVar()
        self.var_w = tk.StringVar(value="640")
        self.var_h = tk.StringVar(value="640")
        self.var_suffix = tk.StringVar(value="")

        self.var_count = tk.StringVar(value="0 / 0")
        self.var_current = tk.StringVar(value="（尚未開始）")

        self.var_anchor = tk.StringVar(value="置中")
        self.var_outfmt = tk.StringVar(value="保持原格式")
        self.var_overwrite = tk.BooleanVar(value=False)

        self.var_jpg_quality = tk.IntVar(value=95)
        self.var_webp_quality = tk.IntVar(value=90)
        self.var_png_compress = tk.IntVar(value=3)

        self.var_color_hex = tk.StringVar(value=bgr_to_hex(self.fill_bgr))

        self._preview_job = None
        self._photo_in = None
        self._photo_out = None

        self._build_ui()
        self._bind_events()

        self.root.after(100, self.update_preview)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        left = ttk.Frame(outer)
        left.pack(side="left", fill="y", padx=(0, 12))

        lf_in = ttk.LabelFrame(left, text="輸入（圖片或資料夾）", padding=10)
        lf_in.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(lf_in)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.var_input).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="選圖片", command=self.pick_input_file).pack(side="left", padx=5)
        ttk.Button(row, text="選資料夾", command=self.pick_input_dir).pack(side="left")

        lf_out = ttk.LabelFrame(left, text="輸出資料夾", padding=10)
        lf_out.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(lf_out)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.var_output).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="瀏覽", command=self.pick_output_dir).pack(side="left", padx=5)

        lf_res = ttk.LabelFrame(left, text="目標解析度（寬 x 高）", padding=10)
        lf_res.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(lf_res)
        row.pack(fill="x")
        ttk.Label(row, text="寬").pack(side="left")
        ttk.Entry(row, textvariable=self.var_w, width=8).pack(side="left", padx=(5, 12))
        ttk.Label(row, text="高").pack(side="left")
        ttk.Entry(row, textvariable=self.var_h, width=8).pack(side="left")

        lf_anchor = ttk.LabelFrame(left, text="定位（貼齊方式）", padding=10)
        lf_anchor.pack(fill="x", pady=(0, 10))
        ttk.Label(lf_anchor, text="縮放後圖片貼到畫布的：").pack(anchor="w")
        ttk.Combobox(lf_anchor, textvariable=self.var_anchor, values=ANCHORS, state="readonly") \
            .pack(fill="x", pady=(6, 0))

        lf_color = ttk.LabelFrame(left, text="填滿顏色（剩餘空白）", padding=10)
        lf_color.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(lf_color)
        row.pack(fill="x")
        self.color_swatch = tk.Canvas(row, width=34, height=18, bd=1, relief="solid")
        self.color_swatch.pack(side="left")
        ttk.Entry(row, textvariable=self.var_color_hex, width=10).pack(side="left", padx=8)
        ttk.Button(row, text="調色盤…", command=self.choose_color_builtin).pack(side="left")
        ttk.Label(lf_color, text="色碼格式：#RRGGBB").pack(anchor="w", pady=(6, 0))
        self._update_color_swatch_from_hex()

        lf_suffix = ttk.LabelFrame(left, text="檔名後綴（空=不加）", padding=10)
        lf_suffix.pack(fill="x", pady=(0, 10))
        ttk.Entry(lf_suffix, textvariable=self.var_suffix).pack(fill="x")

        lf_fmt = ttk.LabelFrame(left, text="輸出格式與品質", padding=10)
        lf_fmt.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(lf_fmt)
        row.pack(fill="x")
        ttk.Label(row, text="輸出格式：").pack(side="left")
        ttk.Combobox(
            row, textvariable=self.var_outfmt,
            values=["保持原格式", "PNG", "JPG", "WEBP"],
            state="readonly", width=10
        ).pack(side="left", padx=6)

        self.frm_q = ttk.Frame(lf_fmt)
        self.frm_q.pack(fill="x", pady=(10, 0))

        self.row_jpg = ttk.Frame(self.frm_q)
        ttk.Label(self.row_jpg, text="JPG 品質：").pack(side="left")
        ttk.Scale(self.row_jpg, from_=0, to=100, variable=self.var_jpg_quality, orient="horizontal") \
            .pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_jpg = ttk.Label(self.row_jpg, text=str(self.var_jpg_quality.get()))
        self.lbl_jpg.pack(side="left")

        self.row_webp = ttk.Frame(self.frm_q)
        ttk.Label(self.row_webp, text="WEBP 品質：").pack(side="left")
        ttk.Scale(self.row_webp, from_=0, to=100, variable=self.var_webp_quality, orient="horizontal") \
            .pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_webp = ttk.Label(self.row_webp, text=str(self.var_webp_quality.get()))
        self.lbl_webp.pack(side="left")

        self.row_png = ttk.Frame(self.frm_q)
        ttk.Label(self.row_png, text="PNG 壓縮：").pack(side="left")
        ttk.Scale(self.row_png, from_=0, to=9, variable=self.var_png_compress, orient="horizontal") \
            .pack(side="left", fill="x", expand=True, padx=6)
        self.lbl_png = ttk.Label(self.row_png, text=str(self.var_png_compress.get()))
        self.lbl_png.pack(side="left")

        ttk.Checkbutton(
            lf_fmt,
            text="同名檔案直接覆蓋（不勾選則自動改名 (1)(2)…）",
            variable=self.var_overwrite
        ).pack(anchor="w", pady=(10, 0))

        lf_prog = ttk.LabelFrame(left, text="進度", padding=10)
        lf_prog.pack(fill="x", pady=(0, 10))
        row = ttk.Frame(lf_prog)
        row.pack(fill="x")
        ttk.Label(row, text="已處理/總件數：").pack(side="left")
        ttk.Label(row, textvariable=self.var_count).pack(side="left")
        ttk.Label(lf_prog, text="目前檔案：").pack(anchor="w", pady=(8, 0))
        ttk.Label(lf_prog, textvariable=self.var_current).pack(fill="x", pady=(2, 0))
        self.pbar = ttk.Progressbar(lf_prog, mode="determinate")
        self.pbar.pack(fill="x", pady=(10, 0))

        btn_row = ttk.Frame(left)
        btn_row.pack(fill="x", pady=(6, 0))
        self.btn_start = ttk.Button(btn_row, text="開始處理", command=self.start)
        self.btn_start.pack(side="left", fill="x", expand=True)
        self.btn_stop = ttk.Button(btn_row, text="停止", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(8, 0))

        right = ttk.Frame(outer)
        right.pack(side="left", fill="both", expand=True)

        lf_prev = ttk.LabelFrame(right, text="8) 預覽（左：原圖 / 右：處理後）", padding=10)
        lf_prev.pack(fill="both", expand=True)

        top = ttk.Frame(lf_prev)
        top.pack(fill="both", expand=True)

        self.prev_in_frame = ttk.LabelFrame(top, text="原圖", padding=8)
        self.prev_in_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.lbl_prev_in = ttk.Label(self.prev_in_frame, anchor="center")
        self.lbl_prev_in.pack(fill="both", expand=True)

        self.prev_out_frame = ttk.LabelFrame(top, text="處理後", padding=8)
        self.prev_out_frame.pack(side="left", fill="both", expand=True)
        self.lbl_prev_out = ttk.Label(self.prev_out_frame, anchor="center")
        self.lbl_prev_out.pack(fill="both", expand=True)

        self.preview_info = ttk.Label(lf_prev, text="提示：改解析度/補色/貼齊/輸出格式會更新預覽（取第一張圖）。")
        self.preview_info.pack(fill="x", pady=(10, 0))

        def upd_labels(*_):
            self.lbl_jpg.config(text=str(int(self.var_jpg_quality.get())))
            self.lbl_webp.config(text=str(int(self.var_webp_quality.get())))
            self.lbl_png.config(text=str(int(self.var_png_compress.get())))
        self.var_jpg_quality.trace_add("write", upd_labels)
        self.var_webp_quality.trace_add("write", upd_labels)
        self.var_png_compress.trace_add("write", upd_labels)

        self._refresh_quality_visibility()

    def _bind_events(self):
        for var in (self.var_input, self.var_output, self.var_w, self.var_h,
                    self.var_suffix, self.var_anchor, self.var_outfmt):
            var.trace_add("write", lambda *_: self.schedule_preview_update())

        self.var_jpg_quality.trace_add("write", lambda *_: self.schedule_preview_update())
        self.var_webp_quality.trace_add("write", lambda *_: self.schedule_preview_update())
        self.var_png_compress.trace_add("write", lambda *_: self.schedule_preview_update())

        self.var_color_hex.trace_add("write", lambda *_: self.on_color_hex_change())

    def schedule_preview_update(self):
        self._refresh_quality_visibility()
        if self._preview_job is not None:
            self.root.after_cancel(self._preview_job)
        self._preview_job = self.root.after(250, self.update_preview)

    def _refresh_quality_visibility(self):
        fmt = self.var_outfmt.get()
        for row in (self.row_jpg, self.row_webp, self.row_png):
            row.pack_forget()
        if fmt == "JPG":
            self.row_jpg.pack(fill="x", pady=(0, 6))
        elif fmt == "WEBP":
            self.row_webp.pack(fill="x", pady=(0, 6))
        elif fmt == "PNG":
            self.row_png.pack(fill="x", pady=(0, 6))

    def pick_input_file(self):
        initdir = initial_dir_from(self.var_input.get())
        path = filedialog.askopenfilename(
            parent=self.root,
            initialdir=initdir,
            title="選擇圖片",
            filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All", "*.*")],
        )
        if path:
            self.var_input.set(path)

    def pick_input_dir(self):
        initdir = initial_dir_from(self.var_input.get())
        path = filedialog.askdirectory(parent=self.root, initialdir=initdir, title="選擇圖片資料夾")
        if path:
            self.var_input.set(path)

    def pick_output_dir(self):
        initdir = initial_dir_from(self.var_output.get())
        path = filedialog.askdirectory(parent=self.root, initialdir=initdir, title="選擇輸出資料夾")
        if path:
            self.var_output.set(path)

    def choose_color_builtin(self):
        _win_center_dialog_over_root_async(self.root, timeout_sec=2.0)
        rgb, hx = colorchooser.askcolor(parent=self.root, title="選擇填滿顏色", color=self.var_color_hex.get())
        if hx:
            self.var_color_hex.set(hx.upper())

    def _update_color_swatch_from_hex(self):
        hx = self.var_color_hex.get().strip().upper()
        try:
            bgr = hex_to_bgr(hx)
        except Exception:
            self.color_swatch.configure(bg="#FFFFFF")
            return
        self.fill_bgr = bgr
        self.color_swatch.configure(bg=bgr_to_hex(bgr))

    def on_color_hex_change(self):
        self._update_color_swatch_from_hex()
        self.schedule_preview_update()

    def _get_first_image_path(self, input_path: str):
        if not input_path:
            return None
        if os.path.isfile(input_path) and is_image_file(input_path):
            return input_path
        if os.path.isdir(input_path):
            files = [os.path.join(input_path, fn) for fn in os.listdir(input_path) if is_image_file(fn)]
            files.sort()
            return files[0] if files else None
        return None

    def _fit_to_label(self, pil_img: Image.Image, label: ttk.Label):
        max_w = max(260, label.winfo_width() - 10)
        max_h = max(260, label.winfo_height() - 10)
        out = pil_img.copy()
        out.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return out

    def update_preview(self):
        self._preview_job = None
        first = self._get_first_image_path(self.var_input.get().strip())
        if not first:
            self.lbl_prev_in.configure(image="", text="（尚未選擇有效的圖片或資料夾）")
            self.lbl_prev_out.configure(image="", text="")
            self._photo_in = None
            self._photo_out = None
            return

        try:
            tw = int(self.var_w.get())
            th = int(self.var_h.get())
            if tw <= 0 or th <= 0:
                return
        except Exception:
            return

        img = cv_imread_unicode(first)
        if img is None:
            self.lbl_prev_in.configure(image="", text="（無法讀取圖片）")
            self.lbl_prev_out.configure(image="", text="")
            self._photo_in = None
            self._photo_out = None
            return

        rgb_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_in = Image.fromarray(rgb_in)
        self._photo_in = ImageTk.PhotoImage(self._fit_to_label(pil_in, self.lbl_prev_in))
        self.lbl_prev_in.configure(image=self._photo_in, text="")

        out = resize_letterbox(img, tw, th, fill_bgr=self.fill_bgr, anchor=self.var_anchor.get())
        rgb_out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(rgb_out)
        self._photo_out = ImageTk.PhotoImage(self._fit_to_label(pil_out, self.lbl_prev_out))
        self.lbl_prev_out.configure(image=self._photo_out, text="")

        self.preview_info.configure(
            text=f"預覽：{os.path.basename(first)} | 目標 {tw}x{th} | 貼齊 {self.var_anchor.get()} | 補色 {bgr_to_hex(self.fill_bgr)}"
        )

    def collect_input_paths(self, input_path: str):
        if os.path.isfile(input_path):
            return [input_path] if is_image_file(input_path) else []
        if os.path.isdir(input_path):
            paths = [os.path.join(input_path, fn) for fn in os.listdir(input_path) if is_image_file(fn)]
            paths.sort()
            return paths
        return []

    def get_output_ext_and_params(self, src_path: str):
        fmt = self.var_outfmt.get()

        if fmt == "保持原格式":
            ext = os.path.splitext(src_path)[1].lower()
            if ext == ".jpeg":
                ext = ".jpg"
            if ext not in [".jpg", ".png", ".webp", ".bmp", ".tif", ".tiff"]:
                ext = ".png"
            params = []
            if ext == ".jpg":
                params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.var_jpg_quality.get())]
            elif ext == ".webp":
                params = [int(cv2.IMWRITE_WEBP_QUALITY), int(self.var_webp_quality.get())]
            elif ext == ".png":
                params = [int(cv2.IMWRITE_PNG_COMPRESSION), int(self.var_png_compress.get())]
            return ext, params

        if fmt == "PNG":
            return ".png", [int(cv2.IMWRITE_PNG_COMPRESSION), int(self.var_png_compress.get())]
        if fmt == "JPG":
            return ".jpg", [int(cv2.IMWRITE_JPEG_QUALITY), int(self.var_jpg_quality.get())]
        return ".webp", [int(cv2.IMWRITE_WEBP_QUALITY), int(self.var_webp_quality.get())]

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("進行中", "目前正在處理中。", parent=self.root)
            return

        input_path = self.var_input.get().strip()
        output_dir = self.var_output.get().strip()

        if not input_path:
            messagebox.showwarning("缺少輸入", "請選擇輸入圖片或資料夾。", parent=self.root)
            return
        if not output_dir:
            messagebox.showwarning("缺少輸出", "請選擇輸出資料夾。", parent=self.root)
            return

        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
            except Exception as e:
                messagebox.showerror("建立輸出資料夾失敗", str(e), parent=self.root)
                return

        try:
            tw = int(self.var_w.get())
            th = int(self.var_h.get())
            if tw <= 0 or th <= 0:
                raise ValueError
        except Exception:
            messagebox.showwarning("解析度錯誤", "請輸入正整數的寬與高。", parent=self.root)
            return

        try:
            _ = hex_to_bgr(self.var_color_hex.get().strip())
        except Exception:
            messagebox.showwarning("顏色錯誤", "填滿顏色請輸入正確的 #RRGGBB。", parent=self.root)
            return

        paths = self.collect_input_paths(input_path)
        if not paths:
            messagebox.showwarning("沒有圖片", "找不到支援的圖片檔。", parent=self.root)
            return

        suffix = self.var_suffix.get()
        anchor = self.var_anchor.get()
        overwrite = bool(self.var_overwrite.get())
        fill_bgr = self.fill_bgr

        self.stop_flag = False
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.pbar.configure(value=0, maximum=len(paths))
        self.var_count.set(f"0 / {len(paths)}")
        self.var_current.set("（準備開始…）")

        self.worker_thread = threading.Thread(
            target=self._worker,
            args=(paths, output_dir, tw, th, suffix, fill_bgr, anchor, overwrite),
            daemon=True,
        )
        self.worker_thread.start()

    def stop(self):
        self.stop_flag = True

    def _safe_set_progress(self, done, total, current_name=None):
        self.pbar.configure(value=done, maximum=total)
        self.var_count.set(f"{done} / {total}")
        if current_name is not None:
            self.var_current.set(current_name)

    def _worker(self, paths, output_dir, tw, th, suffix, fill_bgr, anchor, overwrite):
        total = len(paths)
        done = 0
        failed = 0

        for pth in paths:
            if self.stop_flag:
                break

            base = os.path.basename(pth)
            self.root.after(0, self._safe_set_progress, done, total, base)

            img = cv_imread_unicode(pth)
            if img is None:
                failed += 1
                done += 1
                self.root.after(0, self._safe_set_progress, done, total, f"{base}（讀取失敗）")
                continue

            out = resize_letterbox(img, tw, th, fill_bgr=fill_bgr, anchor=anchor)
            if out is None:
                failed += 1
                done += 1
                self.root.after(0, self._safe_set_progress, done, total, f"{base}（處理失敗）")
                continue

            ext, params = self.get_output_ext_and_params(pth)
            name_no_ext = os.path.splitext(base)[0]
            new_name = f"{name_no_ext}{suffix}{ext}" if suffix else f"{name_no_ext}{ext}"
            save_path = os.path.join(output_dir, new_name)
            if not overwrite:
                save_path = ensure_unique_path(save_path)

            ok = cv_imwrite_unicode(save_path, out, ext=ext, params=params)
            if not ok:
                failed += 1

            done += 1
            self.root.after(0, self._safe_set_progress, done, total, base)

        def finish():
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            if self.stop_flag:
                messagebox.showinfo("已停止", f"已停止處理：{done}/{total}，失敗 {failed}。", parent=self.root)
                self.var_current.set("（已停止）")
            else:
                messagebox.showinfo("完成", f"處理完成：{done}/{total}，失敗 {failed}。\n輸出：{output_dir}", parent=self.root)
                self.var_current.set("（完成）")

        self.root.after(0, finish)


def main():
    root = tk.Tk()

    # Windows DPI（可有可無）
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    ImageResizerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
