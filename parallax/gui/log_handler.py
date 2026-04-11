import logging

from PyQt6.QtCore import QObject, pyqtSignal


class _SignalEmitter(QObject):
    message_logged = pyqtSignal(str)


class SessionLogHandler(logging.Handler):
    """Logging handler that emits Qt signals for each record."""

    def __init__(self):
        super().__init__()
        self._emitter = _SignalEmitter()

    @property
    def message_logged(self):
        return self._emitter.message_logged

    def emit(self, record):
        msg = f"{record.levelname} {self.format(record)}"
        self._emitter.message_logged.emit(msg)
