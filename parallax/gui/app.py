import logging

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QDockWidget, QMainWindow, QMenuBar, QMessageBox,
)

import parallax as par

from parallax.gui.log_handler import SessionLogHandler
from parallax.gui.widgets.toolbar import ParallaxToolbar
from parallax.gui.widgets.log_bar import LogBar
from parallax.gui.panels.sky import SkyPanel
from parallax.gui.panels.detail import DetailPanel
from parallax.gui.panels.reports import ReportsPanel
from parallax.gui.panels.settings import SettingsPanel
from parallax.gui.panels.inspect import InspectWindow

def _normalize_target(name: str) -> str:
    if not name:
        return name
    prefixes = ["Ngc", "Ic", "Hd", "Hr", "Bd", "Pgc", "Ugc", "Mcg"]
    parts = name.strip().split()
    if parts and parts[0] in prefixes:
        parts[0] = parts[0].upper()
    elif parts:
        lower_prefixes = [p.lower() for p in prefixes]
        if parts[0].lower() in lower_prefixes:
            parts[0] = parts[0].upper()
    return " ".join(parts)


_STAGE_PCT = {
    "acquire": 10,
    "downloading": 5,
    "detect": None,   # computed dynamically
    "merge": 65,
    "resolve": 80,
    "report": 88,
    "chart": 93,
    "cutout": 97,
}


class RunWorker(QThread):
    progress = pyqtSignal(str, str)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, target):
        super().__init__()
        self.target = target

    def run(self):
        try:
            report = par.survey.reduce(
                self.target,
                on_progress=lambda step, detail: self.progress.emit(step, detail),
            )
            self.finished.emit(report)
        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._worker = None
        self._detect_count = 0
        self._inspect_windows = []

        self.setDockOptions(
            QMainWindow.DockOption.AllowTabbedDocks | QMainWindow.DockOption.AnimatedDocks
        )

        self._toolbar = ParallaxToolbar()
        self._toolbar.setObjectName("ParallaxToolbar")
        self.addToolBar(self._toolbar)

        self._log_handler = SessionLogHandler()
        logging.getLogger("parallax").addHandler(self._log_handler)

        self._sky = SkyPanel()
        self.setCentralWidget(self._sky)

        self._reports = ReportsPanel()
        self._reports_dock = QDockWidget("Reports", self)
        self._reports_dock.setObjectName("Reports")
        self._reports_dock.setWidget(self._reports)
        self._reports_dock.setMinimumWidth(220)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._reports_dock)

        self._detail = DetailPanel()
        self._detail_dock = QDockWidget("Detail", self)
        self._detail_dock.setObjectName("Detail")
        self._detail_dock.setWidget(self._detail)
        self._detail_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._detail_dock)

        self._log_bar = LogBar()
        self._log_dock = QDockWidget("Log", self)
        self._log_dock.setObjectName("Log")
        self._log_dock.setWidget(self._log_bar)
        self._log_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._log_dock)

        self._log_handler.message_logged.connect(self._log_bar.append)

        self._settings = SettingsPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._settings)
        self._settings.hide()

        self.resizeDocks([self._log_dock], [120], Qt.Orientation.Vertical)

        menu_bar = self.menuBar()
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction(self._reports_dock.toggleViewAction())
        view_menu.addAction(self._detail_dock.toggleViewAction())
        view_menu.addAction(self._log_dock.toggleViewAction())
        view_menu.addAction(self._settings.toggleViewAction())

        from PyQt6.QtCore import QSettings
        settings = QSettings("Parallax", "Parallax")
        geometry = settings.value("mainwindow/geometry_v1")
        state = settings.value("mainwindow/state_v1")
        if geometry:
            self.restoreGeometry(geometry)
        if state:
            self.restoreState(state)

        self._toolbar.run_requested.connect(self._on_run)
        self._toolbar.toggle_known.connect(self._sky.set_known_visible)
        self._toolbar.settings_requested.connect(self._show_settings)
        self._sky.candidate_selected.connect(self._detail.load)
        self._sky.candidate_deselected.connect(self._detail.show_hint)
        self._detail.candidate_closed.connect(self._sky.deselect)
        self._sky.candidate_inspected.connect(self._on_candidate_inspected)
        self._detail.candidate_updated.connect(self._sky.refresh_overlays)
        self._toolbar.search_requested.connect(self._on_search)
        self._reports.report_selected.connect(self._on_report_selected)
        self._reports.report_deleted.connect(self._on_report_deleted)

    def closeEvent(self, event):
        logging.getLogger("parallax").removeHandler(self._log_handler)

        from PyQt6.QtCore import QSettings
        settings = QSettings("Parallax", "Parallax")
        settings.setValue("mainwindow/geometry_v1", self.saveGeometry())
        settings.setValue("mainwindow/state_v1", self.saveState())

        super().closeEvent(event)

    def _on_run(self, target):
        target = _normalize_target(target.strip())
        self._toolbar.set_running(True)
        self._detect_count = 0
        self._sky.show_progress("Starting", target, 0)
        self._worker = RunWorker(target)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_run_finished)
        self._worker.failed.connect(self._on_run_failed)
        self._worker.start()

    def _on_progress(self, step, detail):
        self._toolbar.set_progress(step, detail)
        self._log_bar.append(f"{step}: {detail}")

        if step == "detect":
            # spread detect from 20 to 60
            self._detect_count += 1
            pct = min(60, 20 + self._detect_count * 8)
        else:
            pct = _STAGE_PCT.get(step, 50)

        self._sky.show_progress(step, detail, pct)

    def _on_run_finished(self, report):
        self._toolbar.set_running(False)
        self._toolbar.clear_progress()
        self._reports.refresh()
        self._sky.load_report(report)
        self._sky.set_known_visible(self._toolbar.known_checkbox.isChecked())
        self._detail.show_hint()

    def _on_run_failed(self, message):
        self._toolbar.set_running(False)
        self._toolbar.clear_progress()
        self._sky.show_idle()
        self._detail.show_idle()
        QMessageBox.warning(self, "Run failed", message)

    def _on_report_selected(self, report_id):
        report = par.archive.get_report(report_id)
        if report:
            self._sky.load_report(report)
            self._sky.set_known_visible(self._toolbar.known_checkbox.isChecked())
            self._detail.show_hint()

    def _on_candidate_inspected(self, candidate_id):
        win = InspectWindow(candidate_id, parent=None)
        self._inspect_windows.append(win)
        if len(self._inspect_windows) > 3:
            old = self._inspect_windows.pop(0)
            try:
                old.close()
            except RuntimeError:
                pass  # C++ object already deleted via deleteLater
        win.show()

    def _show_settings(self):
        self._settings.show()
        self._settings.raise_()

    def _on_search(self, query):
        cand = par.catalog.get(query)
        results = None
        if cand is None:
            results = par.archive.search_candidates(query)
            cand = results[0] if results else None

        if cand is None:
            self._toolbar.set_progress("Search", "Not found")
            QTimer.singleShot(2000, self._toolbar.clear_progress)
            return

        if self._sky._current_report_id != cand.report_id:
            report = par.archive.get_report(cand.report_id)
            if report:
                self._sky.load_report(report)
                self._sky.set_known_visible(self._toolbar.known_checkbox.isChecked())

        self._sky.select_candidate(cand.id)
        self._detail.load(cand.id)

        if results and len(results) > 1:
            self._toolbar.set_progress("Search", f"{len(results)} results")

    def _on_report_deleted(self, report_id):
        if self._sky._current_report_id == report_id:
            self._sky.show_idle()
            self._detail.show_idle()


def launch():
    import sys
    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("Parallax")
    window.resize(1200, 800)
    window.show()
    sys.exit(app.exec())
