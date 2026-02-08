from __future__ import annotations

import queue
import time
from typing import Any, Dict, Optional

from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QMessageBox

from app_services import APP_DIR, append_history, now_str
from trainer_worker import TrainConfig, TrainerWorker


class TrainController:
    """Handle training flow and queue-driven UI updates."""

    def __init__(self, app: Any):
        self.app = app

    def parse_positive_int(self, text: str, name: str) -> Optional[int]:
        """Parse a positive integer or show an error."""
        try:
            value = int(text)
        except Exception:
            self.app._show_error("無效參數", f"{name} 必須是整數")
            return None
        if value <= 0:
            self.app._show_error("無效參數", f"{name} 必須是大於 0")
            return None
        return value

    def format_eta_seconds(self, seconds: Optional[float], decimal_places: int = 1) -> str:
        """Format ETA seconds into H:M:S string."""
        if seconds is None:
            return ""
        sec_f = max(0.0, float(seconds))
        sec_i = int(sec_f)
        unit = 10**decimal_places if decimal_places > 0 else 1
        frac_i = int(round((sec_f - sec_i) * unit)) if decimal_places > 0 else 0
        if decimal_places > 0 and frac_i >= unit:
            sec_i += 1
            frac_i = 0

        mins, sec = divmod(sec_i, 60)
        hrs, minute = divmod(mins, 60)
        days, hour = divmod(hrs, 24)
        months, day = divmod(days, 30)
        years, month = divmod(months, 12)
        date_part = f"{years:04d}:{month:02d}:{day:02d} " if (years or month or day) else ""
        clock = f"{hour:02d}:{minute:02d}:{sec:02d}"
        if decimal_places <= 0:
            return f"{date_part}{clock}"
        return f"{date_part}{clock}.{frac_i:0{decimal_places}d}"

    def set_epoch_progress(self, cur: int, total: int, eta_seconds: Optional[float] = None) -> None:
        """Update epoch progress widgets."""
        total = max(1, int(total))
        cur = max(0, min(int(cur), total))
        self.app._last_epoch_progress = (cur, total)
        self.app.pbar_epoch.setMaximum(total)
        self.app.pbar_epoch.setValue(cur)
        if eta_seconds is not None and not (eta_seconds <= 0 and cur >= total):
            self.app._eta_epoch_sec = max(0.0, float(eta_seconds))
            self.app._eta_epoch_last_update = time.time()
        eta_txt = (
            " 剩餘時間：--:--:--"
            if self.app._eta_epoch_sec is None
            else f" 剩餘時間：{self.format_eta_seconds(self.app._eta_epoch_sec, self.app._eta_decimal_places)}"
        )
        self.app.lbl_ep_text.setText(f"Epoch: {cur}/{total}{eta_txt}")

    def set_batch_progress(self, cur: int, total: int, eta_seconds: Optional[float] = None) -> None:
        """Update batch progress widgets."""
        total = max(1, int(total))
        cur = max(0, min(int(cur), total))
        self.app._last_batch_progress = (cur, total)
        self.app.pbar_batch.setMaximum(total)
        self.app.pbar_batch.setValue(cur)
        if eta_seconds is not None and not (eta_seconds <= 0 and cur >= total):
            self.app._eta_batch_sec = max(0.0, float(eta_seconds))
            self.app._eta_batch_last_update = time.time()
        eta_txt = (
            " 剩餘時間：--:--:--"
            if self.app._eta_batch_sec is None
            else f" 剩餘時間：{self.format_eta_seconds(self.app._eta_batch_sec, self.app._eta_decimal_places)}"
        )
        self.app.lbl_ba_text.setText(f"Batch: {cur}/{total}{eta_txt}")

    def tick_eta(self) -> None:
        """Decrease ETA values every timer tick."""
        now = time.time()
        if self.app._eta_epoch_sec is not None and self.app._eta_epoch_sec > 0:
            delta = now - self.app._eta_epoch_last_update
            if delta > 0:
                self.app._eta_epoch_sec = max(0.0, self.app._eta_epoch_sec - delta)
                self.app._eta_epoch_last_update = now
                cur, total = self.app._last_epoch_progress
                self.app.lbl_ep_text.setText(
                    f"Epoch: {cur}/{total} 剩餘時間：{self.format_eta_seconds(self.app._eta_epoch_sec, self.app._eta_decimal_places)}"
                )
        if self.app._eta_batch_sec is not None and self.app._eta_batch_sec > 0:
            delta = now - self.app._eta_batch_last_update
            if delta > 0:
                self.app._eta_batch_sec = max(0.0, self.app._eta_batch_sec - delta)
                self.app._eta_batch_last_update = now
                cur, total = self.app._last_batch_progress
                self.app.lbl_ba_text.setText(
                    f"Batch: {cur}/{total} 剩餘時間：{self.format_eta_seconds(self.app._eta_batch_sec, self.app._eta_decimal_places)}"
                )

    def log(self, text: str) -> None:
        """Append text to log widget."""
        self.app.txt_log.moveCursor(QTextCursor.End)
        self.app.txt_log.insertPlainText(text)
        self.app.txt_log.moveCursor(QTextCursor.End)

    def format_prev_row(self, row: Optional[Dict[str, Any]]) -> str:
        """Format previous epoch row details."""
        if not row:
            return "(無)"
        keys = [
            ("GPU_mem", ["GPU_mem", "gpu_mem"]),
            ("box_loss", ["train/box_loss", "box_loss"]),
            ("cls_loss", ["train/cls_loss", "cls_loss"]),
            ("dfl_loss", ["train/dfl_loss", "dfl_loss"]),
            ("Box(P)", ["metrics/precision(B)", "metrics/precision", "precision"]),
            ("R", ["metrics/recall(B)", "metrics/recall", "recall"]),
            ("mAP50", ["metrics/mAP50(B)", "metrics/mAP50", "mAP50"]),
            ("mAP50-95", ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "mAP5095"]),
        ]
        lines = []
        for label, candidates in keys:
            for key in candidates:
                if key in row and row[key] not in ("", None):
                    lines.append(f"{label}: {row[key]}")
                    break
        return "\n".join(lines) if lines else "(無)"

    def format_epoch_summary(self, row: Optional[Dict[str, Any]]) -> str:
        """Format precision/recall/F1/mAP summary."""
        if not row:
            return "(無)"

        def val(keys: list[str]) -> Optional[float]:
            for key in keys:
                try:
                    if key in row and row[key] not in ("", None):
                        return float(row[key])
                except Exception:
                    pass
            return None

        p = val(["metrics/precision(B)", "metrics/precision", "precision"])
        r = val(["metrics/recall(B)", "metrics/recall", "recall"])
        map50 = val(["metrics/mAP50(B)", "metrics/mAP50", "mAP50"])
        map95 = val(["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "mAP5095"])
        f1 = (2 * p * r / (p + r)) if (p is not None and r is not None and (p + r) > 0) else None

        lines = []
        if p is not None:
            lines.append(f"精確率 (Precision):   {p:.4f}")
        if r is not None:
            lines.append(f"召回率 (Recall):      {r:.4f}")
        if f1 is not None:
            lines.append(f"F1 分數 (F1-score):   {f1:.4f}")
        if map50 is not None:
            lines.append(f"mAP50:               {map50:.4f}")
        if map95 is not None:
            lines.append(f"mAP50-95:            {map95:.4f}")
        return "\n".join(lines) if lines else "(無)"

    def update_metrics_text(self) -> None:
        """Refresh metrics text widget."""
        self.app.txt_metrics.setPlainText(self.app._metrics_log or "(無)")
        self.app.txt_metrics.moveCursor(QTextCursor.End)

    def start_train(self) -> None:
        """Validate form and start worker."""
        self.app._save_config()
        dataset_zip = self.app.edt_dataset_zip.text().strip()
        if not dataset_zip:
            self.app._show_error("缺少資料", "請選擇訓練集")
            return

        work_dir = self.app.edt_work_dir.text().strip() or str(APP_DIR / "workdir")
        out_zip_dir = self.app.edt_out_zip_dir.text().strip() or str(APP_DIR / "output_zips")
        use_custom = self.app.chk_use_custom_model.isChecked()
        custom = self.app.edt_custom_model.text().strip()
        if use_custom and not custom:
            self.app._show_error("缺少自訂權重", "請選擇自訂權重")
            return
        model = custom if use_custom else self.app.cmb_model.currentText().strip()
        if not model:
            self.app._show_error("缺少模型", "請選擇模型或指定自訂權重")
            return

        epochs = self.parse_positive_int(self.app.edt_epochs.text().strip(), "epochs")
        if epochs is None:
            return
        imgsz = self.parse_positive_int(self.app.edt_imgsz.text().strip(), "imgsz")
        if imgsz is None:
            return
        batch = self.parse_positive_int(self.app.edt_batch.text().strip(), "batch")
        if batch is None:
            return

        try:
            self.app._eta_decimal_places = int(self.app.edt_eta_dp.text().strip())
        except Exception:
            self.app._show_error("無效參數", "ETA 小數位數必須是整數")
            return

        cfg = TrainConfig(
            task=self.app.cmb_task.currentText().strip() or "detect",
            dataset_zip=dataset_zip,
            work_dir=work_dir,
            model=model,
            model_dir=self.app.edt_model_dir.text().strip(),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.app.edt_device.text().strip(),
            resume=self.app.chk_resume.isChecked(),
            skip_unlabeled=self.app.chk_skip_unlabeled.isChecked(),
            delete_temp=self.app.chk_delete_temp.isChecked(),
            out_zip_dir=out_zip_dir,
        )

        self.app.txt_log.clear()
        self.log(f"[{now_str()}] ===== 開始訓練 =====\n")
        self.log(f"[{now_str()}] task={cfg.task}\n")
        self.log(f"[{now_str()}] dataset.zip={cfg.dataset_zip}\n")
        self.log(f"[{now_str()}] work_dir={cfg.work_dir}\n")
        self.log(f"[{now_str()}] out_zip_dir={cfg.out_zip_dir}\n")
        self.log(f"[{now_str()}] model={cfg.model}\n")
        self.log(f"[{now_str()}] model_dir={cfg.model_dir}\n")
        self.log(
            f"[{now_str()}] epochs={cfg.epochs}, imgsz={cfg.imgsz}, batch={cfg.batch}, device={cfg.device}, resume={cfg.resume}\n\n"
        )

        self.set_epoch_progress(0, max(1, cfg.epochs))
        self.set_batch_progress(0, 1)
        self.app._metrics_log = ""
        self.update_metrics_text()
        self.app.lbl_status.setText("準備中 ...")
        self.app.lbl_run_device.setText("裝置：偵測中")

        self.app.btn_start.setEnabled(False)
        self.app.btn_stop.setEnabled(True)

        self.app.worker = TrainerWorker(cfg, self.app.msg_q)
        self.app.worker.start()

    def stop_train(self) -> None:
        """Request graceful stop."""
        if self.app.worker:
            self.app.worker.request_stop()
            self.log(f"\n[{now_str()}] 已送出停止請求（將在下一個 callback 觸發時停下）\n")
            self.app.lbl_status.setText("停止中 ...")

    def force_stop_train(self) -> None:
        """Request force stop after user confirmation."""
        if not self.app.worker:
            return
        if self.app._force_stop_locked:
            self.app._show_info("強制停止", "目前正在下載/載入模型，強制停止暫不可用。")
            return
        if QMessageBox.question(self.app, "強制停止", "確定要強制停止嗎？") == QMessageBox.Yes:
            self.app.worker.request_force_stop()
            self.log(f"\n[{now_str()}] 已送出強制停止，訓練將立即中止。\n")
            self.app.lbl_status.setText("強制停止中 ...")
            self.app._show_info("強制停止", "強制停止成功！")

    def poll_queue(self) -> None:
        """Drain worker queue into pending list."""
        try:
            while True:
                self.app._pending_msgs.append(self.app.msg_q.get_nowait())
        except queue.Empty:
            pass
        if not self.app._is_resizing:
            self.flush_pending_msgs()

    def flush_pending_msgs(self) -> None:
        """Apply queued worker messages to UI."""
        while self.app._pending_msgs:
            msg = self.app._pending_msgs.pop(0)
            kind = msg[0]
            if kind == "log":
                self.log(msg[1])
            elif kind == "status":
                self.app.lbl_status.setText(msg[1])
            elif kind == "progress_epoch":
                self.set_epoch_progress(msg[1], msg[2], msg[3] if len(msg) > 3 else None)
            elif kind == "progress_batch":
                self.set_batch_progress(msg[1], msg[2], msg[3] if len(msg) > 3 else None)
            elif kind == "device":
                self.app.lbl_run_device.setText(f"裝置：{msg[1]}")
            elif kind == "prev_epoch_metrics":
                row = msg[1]
                self.app._metrics_log = self.format_prev_row(row) + "\n\n" + self.format_epoch_summary(row)
                self.update_metrics_text()
            elif kind == "force_stop_lock":
                self.app._force_stop_locked = bool(msg[1])
            elif kind == "done":
                self.on_done(msg[1], msg[2])

    def on_done(self, ok: bool, payload: Dict[str, Any]) -> None:
        """Handle training completion message."""
        self.app._force_stop_locked = False
        self.app.btn_start.setEnabled(True)
        self.app.btn_stop.setEnabled(False)

        self.app._eta_epoch_sec = 0.0
        self.app._eta_batch_sec = 0.0
        self.app._eta_epoch_last_update = time.time()
        self.app._eta_batch_last_update = time.time()

        ep_cur, ep_total = self.app._last_epoch_progress
        ba_cur, ba_total = self.app._last_batch_progress
        self.app.lbl_ep_text.setText(
            f"Epoch: {ep_cur}/{ep_total} 剩餘時間：{self.format_eta_seconds(0.0, self.app._eta_decimal_places)}"
        )
        self.app.lbl_ba_text.setText(
            f"Batch: {ba_cur}/{ba_total} 剩餘時間：{self.format_eta_seconds(0.0, self.app._eta_decimal_places)}"
        )

        if ok:
            append_history(payload)
            self.app._load_history()
            rows = payload.get("metrics_rows", {}) or {}
            last_row = rows.get("last")
            self.app._metrics_log = self.format_prev_row(last_row) + "\n\n" + self.format_epoch_summary(last_row)
            self.app._metrics_log += (
                f"\n輸出 ZIP 檔案:       {payload.get('out_zip', '')}\n"
                f"是否中途停止:         {payload.get('stopped', False)}"
            )
            self.update_metrics_text()
            self.app.lbl_status.setText("完成")
            self.log(f"\n[{now_str()}] ===== 完成 =====\n")
            run_dir = payload.get("run_dir", "")
            if run_dir:
                try:
                    self.app.hardcore_panel.load_run(run_dir)
                except Exception as exc:
                    self.log(f"[{now_str()}] 硬核視覺化載入失敗：{exc}\n")
            self.app._show_info(
                "訓練結束" if payload.get("stopped", False) else "訓練完成",
                "訓練已停止。" if payload.get("stopped", False) else "訓練完成。",
            )
        else:
            err = payload.get("error", "Unknown error")
            self.app.lbl_status.setText("失敗")
            self.log(f"\n[{now_str()}] ===== 失敗 =====\n{err}\n")
            self.app._show_error("訓練失敗", err)

        self.app._eta_epoch_sec = None
        self.app._eta_batch_sec = None
        self.app.pbar_epoch.setValue(0)
        self.app.pbar_batch.setValue(0)
        self.app.lbl_ep_text.setText("Epoch: 0/0")
        self.app.lbl_ba_text.setText("Batch: 0/0")
