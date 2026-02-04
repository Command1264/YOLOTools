from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


class LogStore:
    def __init__(self, exec_dir: Path) -> None:
        self.exec_dir = exec_dir
        self.log_dir = exec_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        name = f"python_packager_gui_{datetime.now().strftime('%Y%m%d')}.log"
        self.log_path = self.log_dir / name
        self.logger = logging.getLogger("python_packager_gui")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            handler = logging.FileHandler(self.log_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
