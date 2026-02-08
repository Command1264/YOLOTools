from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from main_window import App


def main() -> int:
    """Run YOLO trainer GUI."""
    app = QApplication(sys.argv)
    window = App()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
