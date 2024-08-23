import os
import sys
import pytest

PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    try:
        _app = QApplication([])
    except Exception:
        pytest.skip("no display available", allow_module_level=True)


def test_launch_callable():
    from parallax.gui import launch
    assert callable(launch)


def test_mainwindow_instantiates(tmp_db):
    from parallax.gui.app import MainWindow
    w = MainWindow()
    assert w is not None
    w.close()


def test_reports_panel(tmp_db):
    from parallax.gui.panels.reports import ReportsPanel
    panel = ReportsPanel()
    panel.refresh()
    panel.close()


def test_sky_panel_clear():
    from parallax.gui.panels.sky import SkyPanel
    panel = SkyPanel()
    panel.clear()
    panel.close()


def test_detail_panel_clear():
    from parallax.gui.panels.detail import DetailPanel
    panel = DetailPanel()
    panel.clear()
    panel.close()


def test_run_worker_init():
    from parallax.gui.app import RunWorker
    worker = RunWorker("M92")
    assert worker.target == "M92"


def test_session_log_handler_emits_signal():
    import logging
    from parallax.gui.log_handler import SessionLogHandler

    handler = SessionLogHandler()
    received = []
    handler.message_logged.connect(received.append)

    logger = logging.getLogger("parallax.test.handler")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.info("hello from test")
    logger.removeHandler(handler)

    assert len(received) == 1
    assert received[0] == "INFO hello from test"


def test_session_log_handler_format():
    import logging
    from parallax.gui.log_handler import SessionLogHandler

    handler = SessionLogHandler()
    msgs = []
    handler.message_logged.connect(msgs.append)

    logger = logging.getLogger("parallax.test.fmt")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.warning("something broke")
    logger.debug("trace detail")
    logger.removeHandler(handler)

    assert msgs[0] == "WARNING something broke"
    assert msgs[1] == "DEBUG trace detail"
