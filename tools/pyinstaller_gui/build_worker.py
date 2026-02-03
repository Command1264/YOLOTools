from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PySide6 import QtCore


@dataclass
class BuildSettings:
    packager: str
    script_path: str
    name: str
    onefile: bool
    no_console: bool
    icon_path: str
    add_data: List[str]
    hidden_imports: List[str]
    work_path: str
    dist_path: str
    spec_path: str
    clean: bool
    extra_args: str
    strip: bool
    upx: bool
    upx_dir: str
    debug: bool
    runtime_tmp: str
    lto: bool
    plugins: str


class BuildWorker(QtCore.QThread):
    output_line = QtCore.Signal(str)
    finished_ok = QtCore.Signal(bool)

    def __init__(self, settings: BuildSettings, parent=None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._proc: Optional[subprocess.Popen] = None
        self._stop_requested = False
        self._stdin_lock = QtCore.QMutex()

    def run(self) -> None:
        cmd = self._build_command()
        self.output_line.emit(" ".join(cmd))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        except Exception as exc:
            self.output_line.emit(f"[ERROR] 無法啟動 PyInstaller：{exc}")
            self.finished_ok.emit(False)
            return

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self.output_line.emit(line.rstrip())
            if self._stop_requested:
                break

        if self._stop_requested and self._proc:
            try:
                self._proc.terminate()
            except Exception:
                pass

        code = self._proc.wait() if self._proc else 1
        self.finished_ok.emit(code == 0 and not self._stop_requested)

    def stop(self) -> None:
        self._stop_requested = True
        if self._proc:
            try:
                self._terminate_process_tree()
            except Exception:
                pass

    def send_input(self, text: str) -> None:
        if not self._proc or self._proc.stdin is None:
            return
        with QtCore.QMutexLocker(self._stdin_lock):
            try:
                self._proc.stdin.write(text + "\n")
                self._proc.stdin.flush()
            except Exception:
                pass

    def _terminate_process_tree(self) -> None:
        if not self._proc:
            return
        try:
            import psutil
        except Exception:
            self._proc.terminate()
            return
        try:
            proc = psutil.Process(self._proc.pid)
            children = proc.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except Exception:
                    pass
            try:
                proc.terminate()
            except Exception:
                pass
            psutil.wait_procs([proc, *children], timeout=3)
        except Exception:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _build_command(self) -> List[str]:
        if self.settings.packager == "nuitka":
            return self._build_nuitka_command()
        return self._build_pyinstaller_command()

    def _build_pyinstaller_command(self) -> List[str]:
        sep = ";" if os.name == "nt" else ":"
        s = self.settings
        cmd = ["pyinstaller"]
        if s.onefile:
            cmd.append("--onefile")
        if s.no_console:
            cmd.append("--noconsole")
        if s.clean:
            cmd.append("--clean")
        if s.name:
            cmd.extend(["--name", s.name])
        if s.icon_path:
            cmd.extend(["--icon", s.icon_path])
        if s.work_path:
            cmd.extend(["--workpath", s.work_path])
        if s.dist_path:
            cmd.extend(["--distpath", s.dist_path])
        if s.spec_path:
            cmd.extend(["--specpath", s.spec_path])
        if s.strip:
            cmd.append("--strip")
        if s.upx:
            cmd.append("--upx")
        if s.upx_dir:
            cmd.extend(["--upx-dir", s.upx_dir])
        if s.debug:
            cmd.extend(["--debug", "all"])
        if s.runtime_tmp:
            cmd.extend(["--runtime-tmpdir", s.runtime_tmp])
        for item in s.add_data:
            if item.strip():
                cmd.extend(["--add-data", item.strip().replace("|", sep)])
        for item in s.hidden_imports:
            if item.strip():
                cmd.extend(["--hidden-import", item.strip()])
        if s.extra_args.strip():
            cmd.extend(shlex.split(s.extra_args))
        cmd.append(s.script_path)
        return cmd

    def _build_nuitka_command(self) -> List[str]:
        sep = ";" if os.name == "nt" else ":"
        s = self.settings
        cmd = [sys.executable, "-m", "nuitka", "--mode=onefile" if s.onefile else "--mode=standalone"]
        if s.lto:
            cmd.append("--lto")
        if s.plugins:
            for p in s.plugins.split(","):
                p = p.strip()
                if p:
                    cmd.append(f"--plugin-enable={p}")
        if s.no_console and os.name == "nt":
            cmd.append("--windows-console-mode=disable")
        if s.icon_path and os.name == "nt":
            cmd.append(f"--windows-icon-from-ico={s.icon_path}")
        if s.name:
            cmd.append(f"--output-filename={s.name}")
        if s.dist_path:
            cmd.append(f"--output-dir={s.dist_path}")
        for item in s.add_data:
            if item.strip():
                src, _, dst = item.strip().partition("|")
                if dst:
                    cmd.append(f"--include-data-file={src}{sep}{dst}")
        for item in s.hidden_imports:
            if item.strip():
                cmd.append(f"--include-module={item.strip()}")
        if s.extra_args.strip():
            cmd.extend(shlex.split(s.extra_args))
        cmd.append(s.script_path)
        return cmd
