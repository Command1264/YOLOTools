from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtWidgets import QDialog, QTableWidgetItem

from app_services import HISTORY_PATH, read_history
from dialogs import DeleteHistoryDialog, show_json_record_dialog
from history_model import HistoryRecord


class HistoryController:
    """Handle history tab loading and actions."""

    def __init__(self, app: Any):
        self.app = app

    def load_history(self) -> None:
        """Load rows into history table."""
        items = list(reversed(read_history(800)))
        self.app.tree.setRowCount(len(items))
        for row_idx, rec in enumerate(items):
            met = rec.get("metrics", {}) or {}
            map50 = met.get("mAP50")
            recall = met.get("recall")
            vals = [
                rec.get("time", ""),
                rec.get("task", ""),
                rec.get("model", ""),
                rec.get("dataset_zip", ""),
                "" if map50 is None else f"{map50:.4f}",
                "" if recall is None else f"{recall:.4f}",
                rec.get("out_zip", ""),
                rec.get("run_dir", ""),
            ]
            for col, value in enumerate(vals):
                self.app.tree.setItem(row_idx, col, QTableWidgetItem(str(value)))
        self.app.tree.resizeColumnsToContents()

    def selected_rows(self) -> list[int]:
        """Return selected row indexes."""
        return sorted({item.row() for item in self.app.tree.selectionModel().selectedRows()})

    def selected_record(self) -> Optional[Dict[str, Any]]:
        """Resolve selected table row to full history record."""
        rows = self.selected_rows()
        if not rows:
            return None
        row = rows[0]
        time_val = self.app.tree.item(row, 0).text() if self.app.tree.item(row, 0) else ""
        out_zip = self.app.tree.item(row, 6).text() if self.app.tree.item(row, 6) else ""
        run_dir = self.app.tree.item(row, 7).text() if self.app.tree.item(row, 7) else ""
        for rec in reversed(read_history(3000)):
            if rec.get("time") == time_val and rec.get("out_zip") == out_zip and rec.get("run_dir") == run_dir:
                return rec
        return None

    def ensure_single_selection(self, action_name: str) -> bool:
        """Ensure action operates on exactly one row."""
        if len(self.selected_rows()) > 1:
            self.app._show_warning("多選限制", f"{action_name} 只能單選，請只選擇一筆紀錄。")
            return False
        return True

    def show_selected_detail(self) -> None:
        """Show selected history record details."""
        if not self.ensure_single_selection("查看詳細"):
            return
        rec = self.selected_record()
        if not rec:
            self.app._show_info("提示", "請先選擇一筆歷史紀錄")
            return
        show_json_record_dialog(self.app, "訓練詳細", rec)

    def open_selected_zip_dir(self) -> None:
        """Open selected zip directory or select zip file in explorer."""
        rows = self.selected_rows()
        if not rows:
            self.app._show_info("提示", "請先選擇一筆歷史紀錄")
            return
        if len(rows) == 1:
            rec = self.selected_record()
            if rec:
                zip_path = rec.get("out_zip", "")
                if zip_path:
                    path = Path(zip_path)
                    if path.exists() and os.name == "nt":
                        try:
                            subprocess.Popen(["explorer", f"/select,{path}"])
                            return
                        except Exception:
                            pass

        rec = self.selected_record()
        if not rec:
            self.app._show_info("提示", "請先選擇一筆歷史紀錄")
            return
        zip_path = rec.get("out_zip", "")
        if not zip_path:
            self.app._show_error("找不到歷史紀錄", "找不到歷史紀錄資料夾")
            return
        self.open_folder(Path(zip_path).parent)

    def load_selected_to_hardcore(self) -> None:
        """Load selected run into hardcore panel."""
        if not self.ensure_single_selection("載入硬核視覺化"):
            return
        rec = self.selected_record()
        if not rec:
            self.app._show_info("提示", "請先選擇一筆歷史紀錄")
            return
        run_dir = rec.get("run_dir", "")
        if not run_dir:
            self.app._show_error("載入失敗", "找不到此訓練結果")
            return
        try:
            self.app.hardcore_panel.load_run(run_dir)
            self.app._show_info("載入成功", "成功載入至硬核視覺化")
        except Exception as exc:
            self.app._show_error("載入失敗", f"硬核視覺化載入失敗：\n{exc}")

    def open_selected_zip(self) -> None:
        """Open selected output zip."""
        if not self.ensure_single_selection("開啟輸出 zip"):
            return
        rec = self.selected_record()
        if not rec:
            self.app._show_info("提示", "請先選擇一筆歷史紀錄")
            return
        zip_path = rec.get("out_zip", "")
        if not zip_path:
            self.app._show_warning("找不到輸出", "找不到輸出 zip。")
            return
        path = Path(zip_path)
        if not path.exists():
            self.app._show_warning("找不到輸出", "輸出 zip 不存在。")
            return

        try:
            if os.name == "nt":
                os.startfile(str(path))
                return
        except Exception:
            pass

        try:
            if os.name == "nt":
                subprocess.Popen(["explorer", f"/select,{path}"])
            else:
                self.open_folder(path.parent)
        except Exception:
            self.open_folder(path.parent)

    def delete_selected_history(self) -> None:
        """Delete selected history rows and optional zip files."""
        rows = self.selected_rows()
        if not rows:
            self.app._show_info("提示", "請先選擇要刪除的歷史紀錄")
            return
        dialog = DeleteHistoryDialog(len(rows), self.app)
        if dialog.exec() != QDialog.Accepted:
            return
        del_zip = dialog.chk_delete_zip.isChecked()

        selected_keys = set()
        for row in rows:
            vals = [self.app.tree.item(row, i).text() if self.app.tree.item(row, i) else "" for i in (0, 6, 7)]
            selected_keys.add((vals[0], vals[1], vals[2]))

        items = read_history(100000)
        kept = []
        removed = []
        for rec in items:
            key = (rec.get("time", ""), rec.get("out_zip", ""), rec.get("run_dir", ""))
            if key in selected_keys:
                removed.append(rec)
            else:
                kept.append(rec)

        try:
            HISTORY_PATH.write_text(
                "\n".join(HistoryRecord(data=dict(x)).to_json_line() for x in kept) + ("\n" if kept else ""),
                encoding="utf-8",
            )
        except Exception as exc:
            self.app._show_error("刪除失敗", f"無法更新歷史檔：{exc}")
            return

        if del_zip:
            for rec in removed:
                zip_path = rec.get("out_zip", "")
                if not zip_path:
                    continue
                try:
                    Path(zip_path).unlink(missing_ok=True)
                except Exception:
                    pass

        self.load_history()
        self.app._show_info("完成", f"已刪除 {len(removed)} 筆歷史紀錄。")

    def open_folder(self, folder: Path) -> None:
        """Open folder in platform file explorer."""
        try:
            folder = folder.resolve()
        except Exception:
            pass
        try:
            if os.name == "nt":
                os.startfile(str(folder))
                return
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            self.app._show_error("開啟失敗", f"無法開啟資料夾：\n{exc}")
