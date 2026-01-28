from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Callable, Optional

import tkinter as tk
from tkinter import ttk, messagebox

from log_manager import get_active_log_file, get_log_queue, list_log_files


class LogViewer(tk.Toplevel):
    """提供 GUI 方式瀏覽與即時顯示運行日誌。"""

    def __init__(self, parent: tk.Tk, on_close: Optional[Callable[[], None]] = None) -> None:
        super().__init__(parent)
        self._parent = parent
        self._on_close = on_close
        self._log_queue: queue.Queue[str] = get_log_queue()
        self._files: list[Path] = []
        self._current_file: Optional[Path] = None
        self._live_file: Optional[Path] = get_active_log_file()
        self._loading_thread: Optional[threading.Thread] = None
        self._loading = False

        self.title("運行日誌")
        self.geometry("820x520")
        self.minsize(720, 420)

        self._build_ui()
        self._refresh_files(select_latest=True)
        self._poll_log_queue()

        self.transient(parent)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self._close)

        try:
            if hasattr(parent, "_center_dialog"):
                parent._center_dialog(self)
            else:
                self._center_on_screen()
        except Exception:
            pass

    def _build_ui(self) -> None:
        top_bar = ttk.Frame(self, padding=10)
        top_bar.pack(fill="x")

        ttk.Label(top_bar, text="檔案").pack(side="left")
        self._var_file = tk.StringVar()
        self.cmb_files = ttk.Combobox(top_bar, textvariable=self._var_file, state="readonly", width=40)
        self.cmb_files.pack(side="left", padx=(6, 8))
        self.cmb_files.bind("<<ComboboxSelected>>", self._on_file_selected)

        self.btn_prev = ttk.Button(top_bar, text="上一頁", command=self._prev_file)
        self.btn_prev.pack(side="left", padx=4)
        self.btn_next = ttk.Button(top_bar, text="下一頁", command=self._next_file)
        self.btn_next.pack(side="left", padx=4)
        ttk.Button(top_bar, text="重新整理", command=lambda: self._refresh_files(select_latest=False)).pack(
            side="left", padx=4
        )

        btns = ttk.Frame(top_bar)
        btns.pack(side="right")
        ttk.Button(btns, text="複製選取", command=self._copy_selection).pack(side="right", padx=4)
        ttk.Button(btns, text="複製全部", command=self._copy_all).pack(side="right", padx=4)

        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)

        self.txt_log = tk.Text(body, wrap="none", state="disabled")
        self.txt_log.pack(side="left", fill="both", expand=True)

        y_scroll = ttk.Scrollbar(body, orient="vertical", command=self.txt_log.yview)
        y_scroll.pack(side="right", fill="y")
        self.txt_log.configure(yscrollcommand=y_scroll.set)

        self.txt_log.tag_configure("INFO", foreground="#1f2933")
        self.txt_log.tag_configure("WARNING", foreground="#b45309")
        self.txt_log.tag_configure("ERROR", foreground="#b91c1c")
        self.txt_log.tag_configure("DEBUG", foreground="#6b7280")
        self.txt_log.tag_configure("LOADING", foreground="#6b7280", justify="center")
        self.txt_log.bind("<Configure>", self._on_text_resize)

    def _refresh_files(self, select_latest: bool) -> None:
        self._files = list_log_files()
        display = [p.name for p in self._files]
        self.cmb_files["values"] = display

        if not self._files:
            self._set_text("尚無日誌檔案。")
            self._update_nav_buttons()
            return

        target = None
        if select_latest:
            target = self._files[-1]
        else:
            current = self._current_file
            if current and current in self._files:
                target = current
            else:
                target = self._files[-1]

        if target is not None:
            self._set_current_file(target)
        self._update_nav_buttons()

    def _set_current_file(self, path: Path) -> None:
        self._current_file = path
        self._live_file = get_active_log_file()
        self._var_file.set(path.name)
        self._drain_log_queue()
        self._load_file_async(path)
        self._update_nav_buttons()

    def _load_file(self, path: Path) -> None:
        if not path.exists():
            self._set_text("日誌檔案不存在。")
            return
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            self._set_text("無法讀取日誌檔案。")
            return
        self._set_text("")
        for line in content.splitlines():
            self._append_line(line)
        self._scroll_to_end()

    def _load_file_async(self, path: Path) -> None:
        if self._loading:
            return
        self._set_loading(True)

        def _worker():
            if not path.exists():
                self.after(0, lambda: self._finish_loading("日誌檔案不存在。"))
                return
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                self.after(0, lambda: self._finish_loading("無法讀取日誌檔案。"))
                return
            self.after(0, lambda: self._finish_loading(content))

        self._loading_thread = threading.Thread(target=_worker, daemon=True)
        self._loading_thread.start()

    def _finish_loading(self, content: str) -> None:
        self._set_text("")
        if content in {"日誌檔案不存在。", "無法讀取日誌檔案。"}:
            self._set_text(content)
            self._set_loading(False)
            return
        for line in content.splitlines():
            self._append_line(line)
        self._scroll_to_end()
        self._set_loading(False)

    def _append_line(self, line: str) -> None:
        tag = self._level_tag(line)
        self.txt_log.configure(state="normal")
        self.txt_log.insert("end", line + "\n", tag)
        self.txt_log.configure(state="disabled")

    def _set_text(self, text: str) -> None:
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        if text:
            self.txt_log.insert("end", text)
        self.txt_log.configure(state="disabled")

    def _set_loading(self, loading: bool) -> None:
        self._loading = loading
        if loading:
            self._set_center_text("正在載入中")
        self.btn_prev.configure(state="disabled" if loading else self.btn_prev["state"])
        self.btn_next.configure(state="disabled" if loading else self.btn_next["state"])
        self.cmb_files.configure(state="disabled" if loading else "readonly")

    def _set_center_text(self, text: str) -> None:
        self.txt_log.configure(state="normal")
        self.txt_log.delete("1.0", "end")
        total_lines = self._visible_lines()
        pad_lines = max(0, (total_lines // 2) - 1)
        if pad_lines:
            self.txt_log.insert("1.0", "\n" * pad_lines)
        self.txt_log.insert("end", text, "LOADING")
        self.txt_log.configure(state="disabled")
        self.txt_log.see("1.0")

    def _on_text_resize(self, _event=None) -> None:
        if self._loading:
            self._set_center_text("正在載入中")

    def _visible_lines(self) -> int:
        try:
            height = max(1, self.txt_log.winfo_height())
            line_px = max(1, int(self.txt_log.dlineinfo("1.0")[3]))
            return max(1, height // line_px)
        except Exception:
            return 1

    def _scroll_to_end(self) -> None:
        self.txt_log.see("end")

    def _level_tag(self, line: str) -> str:
        if "[ERROR]" in line:
            return "ERROR"
        if "[WARNING]" in line:
            return "WARNING"
        if "[DEBUG]" in line:
            return "DEBUG"
        return "INFO"

    def _poll_log_queue(self) -> None:
        try:
            while True:
                line = self._log_queue.get_nowait()
                if self._current_file and self._live_file and self._current_file == self._live_file:
                    self._append_line(line)
                    self._scroll_to_end()
        except queue.Empty:
            pass
        self.after(200, self._poll_log_queue)

    def _on_file_selected(self, _event=None) -> None:
        name = self._var_file.get()
        for path in self._files:
            if path.name == name:
                self._set_current_file(path)
                return

    def _prev_file(self) -> None:
        if not self._files:
            return
        if self._current_file not in self._files:
            self._set_current_file(self._files[-1])
            return
        idx = self._files.index(self._current_file)
        if idx <= 0:
            return
        self._set_current_file(self._files[idx - 1])

    def _next_file(self) -> None:
        if not self._files:
            return
        if self._current_file not in self._files:
            self._set_current_file(self._files[-1])
            return
        idx = self._files.index(self._current_file)
        if idx >= len(self._files) - 1:
            return
        self._set_current_file(self._files[idx + 1])

    def _update_nav_buttons(self) -> None:
        if not self._files or self._current_file not in self._files:
            self.btn_prev.configure(state="disabled")
            self.btn_next.configure(state="disabled")
            return
        idx = self._files.index(self._current_file)
        self.btn_prev.configure(state="disabled" if idx <= 0 else "normal")
        self.btn_next.configure(state="disabled" if idx >= len(self._files) - 1 else "normal")

    def _drain_log_queue(self) -> None:
        try:
            while True:
                self._log_queue.get_nowait()
        except queue.Empty:
            pass

    def _copy_all(self) -> None:
        text = self.txt_log.get("1.0", "end-1c")
        self._copy_to_clipboard(text)

    def _copy_selection(self) -> None:
        try:
            text = self.txt_log.get("sel.first", "sel.last")
        except Exception:
            messagebox.showinfo("複製", "請先選取要複製的文字。")
            return
        self._copy_to_clipboard(text)

    def _copy_to_clipboard(self, text: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(text)
        self.update()

    def _center_on_screen(self) -> None:
        self.update_idletasks()
        w = self.winfo_reqwidth()
        h = self.winfo_reqheight()
        x = (self.winfo_screenwidth() - w) // 2
        y = (self.winfo_screenheight() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

    def _close(self) -> None:
        if self._on_close:
            try:
                self._on_close()
            except Exception:
                pass
        self.destroy()
