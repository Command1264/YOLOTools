from __future__ import annotations

from typing import Dict

from PySide6 import QtWidgets


class AdvancedDialog(QtWidgets.QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("進階參數")
        self.setModal(False)
        self.resize(520, 420)
        self._packager = "pyinstaller"

        self.grp_py = QtWidgets.QGroupBox("PyInstaller 進階參數")
        self.chk_strip = QtWidgets.QCheckBox("去除符號表 (--strip)")
        self.chk_upx = QtWidgets.QCheckBox("使用 UPX 壓縮 (--upx)")
        self.txt_upx_dir = QtWidgets.QLineEdit()
        self.txt_upx_dir.setPlaceholderText("UPX 路徑 (可空)")
        self.chk_debug = QtWidgets.QCheckBox("Debug 模式 (--debug all)")
        self.txt_runtime_tmp = QtWidgets.QLineEdit()
        self.txt_runtime_tmp.setPlaceholderText("runtime-tempdir")

        py_form = QtWidgets.QFormLayout()
        py_form.addRow(self.chk_strip)
        py_form.addRow(self.chk_upx)
        py_form.addRow("UPX 路徑", self.txt_upx_dir)
        py_form.addRow(self.chk_debug)
        py_form.addRow("Runtime Temp Dir", self.txt_runtime_tmp)
        self.grp_py.setLayout(py_form)

        self.grp_nuitka = QtWidgets.QGroupBox("Nuitka 進階參數")
        self.chk_lto = QtWidgets.QCheckBox("LTO (--lto)")
        self.txt_plugin = QtWidgets.QLineEdit()
        self.txt_plugin.setPlaceholderText("plugins (以逗號分隔)")

        nk_form = QtWidgets.QFormLayout()
        nk_form.addRow(self.chk_lto)
        nk_form.addRow("Plugins", self.txt_plugin)
        self.grp_nuitka.setLayout(nk_form)

        self.txt_extra = QtWidgets.QLineEdit()
        self.txt_extra.setPlaceholderText("其他參數（原樣傳入）")

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.grp_py)
        layout.addWidget(self.grp_nuitka)
        layout.addWidget(QtWidgets.QLabel("其他參數"))
        layout.addWidget(self.txt_extra)
        layout.addWidget(btns)

        self._apply_packager_ui()

    def set_packager(self, name: str) -> None:
        self._packager = name or "pyinstaller"
        self._apply_packager_ui()

    def _apply_packager_ui(self) -> None:
        is_py = self._packager == "pyinstaller"
        self.grp_py.setVisible(is_py)
        self.grp_nuitka.setVisible(not is_py)

    def get_extra_args(self) -> str:
        return self.txt_extra.text().strip()

    def set_extra_args(self, text: str) -> None:
        self.txt_extra.setText(text or "")

    def to_dict(self) -> Dict[str, object]:
        return {
            "strip": self.chk_strip.isChecked(),
            "upx": self.chk_upx.isChecked(),
            "upx_dir": self.txt_upx_dir.text().strip(),
            "debug": self.chk_debug.isChecked(),
            "runtime_tmp": self.txt_runtime_tmp.text().strip(),
            "lto": self.chk_lto.isChecked(),
            "plugins": self.txt_plugin.text().strip(),
            "extra_args": self.txt_extra.text().strip(),
        }

    def apply_dict(self, data: Dict[str, object]) -> None:
        self.chk_strip.setChecked(bool(data.get("strip", False)))
        self.chk_upx.setChecked(bool(data.get("upx", False)))
        self.txt_upx_dir.setText(str(data.get("upx_dir", "")))
        self.chk_debug.setChecked(bool(data.get("debug", False)))
        self.txt_runtime_tmp.setText(str(data.get("runtime_tmp", "")))
        self.chk_lto.setChecked(bool(data.get("lto", False)))
        self.txt_plugin.setText(str(data.get("plugins", "")))
        self.txt_extra.setText(str(data.get("extra_args", "")))
