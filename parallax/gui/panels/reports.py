import os

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem,
    QMenu, QMessageBox,
)

from parallax.gui.platform import open_folder, open_file, reveal_file


class ReportsPanel(QWidget):
    report_selected = pyqtSignal(str)
    report_deleted = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_double_click)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self._list)

        self._reports_cache = {}  # report_id -> Report
        self.refresh()

    def refresh(self):
        self._list.clear()
        self._reports_cache.clear()
        try:
            from parallax import archive
            reports = archive.reports(limit=50)
        except Exception:
            return

        for rpt in reports:
            date = rpt.created_at.strftime("%Y-%m-%d") if rpt.created_at else "?"
            text = f"{rpt.target}  {date}  {rpt.n_unverified} unverified"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, rpt.id)
            self._list.addItem(item)
            self._reports_cache[rpt.id] = rpt

    def _on_double_click(self, item):
        rid = item.data(Qt.ItemDataRole.UserRole)
        if rid:
            self.report_selected.emit(rid)

    def _show_context_menu(self, pos):
        item = self._list.itemAt(pos)
        if item is None:
            return
        rid = item.data(Qt.ItemDataRole.UserRole)
        rpt = self._reports_cache.get(rid)
        if not rpt:
            return

        menu = QMenu(self)

        open_action = menu.addAction("Open folder")
        view_md = menu.addAction("View markdown")
        export_csv = menu.addAction("Export CSV")
        integrity_action = menu.addAction("Check integrity")
        menu.addSeparator()
        delete_action = menu.addAction("Delete")

        action = menu.exec(self._list.mapToGlobal(pos))
        if action is None:
            return

        if action == open_action:
            self._open_folder(rpt)
        elif action == view_md:
            self._view_markdown(rpt)
        elif action == export_csv:
            self._export_csv(rpt)
        elif action == integrity_action:
            self._check_integrity(rpt)
        elif action == delete_action:
            self._delete_report(rpt)

    def _open_folder(self, rpt):
        path = rpt.md_path or rpt.json_path
        if path and os.path.exists(os.path.dirname(path)):
            open_folder(os.path.dirname(path))

    def _view_markdown(self, rpt):
        if rpt.md_path and os.path.isfile(rpt.md_path):
            open_file(rpt.md_path)

    def _export_csv(self, rpt):
        try:
            from parallax import archive
            csv_path = archive.export(rpt.id, format="csv")
            reveal_file(csv_path)
        except Exception as e:
            QMessageBox.warning(self, "Export failed", str(e))

    def _check_integrity(self, rpt):
        from parallax._db import get_db

        missing = []
        input_paths = []

        try:
            with get_db() as conn:
                rows = conn.execute(
                    "SELECT fits_path FROM report_inputs WHERE report_id = ?",
                    (rpt.id,),
                ).fetchall()
                input_paths = [r["fits_path"] for r in rows]
        except Exception:
            pass

        for p in input_paths:
            if p and not os.path.isfile(p):
                missing.append(p)

        for p in [rpt.json_path, rpt.md_path]:
            if p and not os.path.isfile(p):
                missing.append(p)

        if not missing:
            QMessageBox.information(self, "Integrity", "All files present.")
            return

        msg = "Missing files:\n" + "\n".join(missing)
        msg += "\n\nRemove orphaned input records?"
        reply = QMessageBox.warning(
            self, "Integrity",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        missing_inputs = [p for p in input_paths if p and not os.path.isfile(p)]
        if missing_inputs:
            try:
                with get_db() as conn:
                    for p in missing_inputs:
                        conn.execute(
                            "DELETE FROM report_inputs WHERE report_id = ? AND fits_path = ?",
                            (rpt.id, p),
                        )
            except Exception:
                pass

        self.refresh()

    def _delete_report(self, rpt):
        reply = QMessageBox.question(
            self, "Delete report",
            f"Delete report {rpt.id} and its files?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        report_dir = os.path.dirname(rpt.md_path or rpt.json_path or "")
        if report_dir and os.path.isdir(report_dir):
            for fname in os.listdir(report_dir):
                if fname.startswith(rpt.id):
                    try:
                        os.remove(os.path.join(report_dir, fname))
                    except OSError:
                        pass

        if report_dir and os.path.isdir(report_dir):
            try:
                os.rmdir(report_dir)
                parent = os.path.dirname(report_dir)
                os.rmdir(parent)
            except OSError:
                pass

        try:
            from parallax._db import get_db
            with get_db() as conn:
                conn.execute("DELETE FROM catalog_matches WHERE candidate_id IN "
                             "(SELECT id FROM candidates WHERE report_id = ?)", (rpt.id,))
                conn.execute("DELETE FROM candidate_detections WHERE candidate_id IN "
                             "(SELECT id FROM candidates WHERE report_id = ?)", (rpt.id,))
                conn.execute("DELETE FROM candidates WHERE report_id = ?", (rpt.id,))
                conn.execute("DELETE FROM report_inputs WHERE report_id = ?", (rpt.id,))
                conn.execute("DELETE FROM reports WHERE id = ?", (rpt.id,))
        except Exception:
            pass

        self.report_deleted.emit(rpt.id)
        self.refresh()
