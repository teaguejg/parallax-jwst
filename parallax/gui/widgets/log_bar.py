from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPalette
from PyQt6.QtWidgets import QLabel, QPlainTextEdit, QVBoxLayout, QWidget


class LogBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._history = []

        mono = QFont("Consolas", 9)
        mono.setStyleHint(QFont.StyleHint.Monospace)

        self._label = QLabel()
        self._label.setFont(mono)
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._label.setFixedHeight(22)

        self._console = QPlainTextEdit()
        self._console.setReadOnly(True)
        self._console.setFont(mono)
        self._console.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        layout.addWidget(self._console)

        self._apply_style()

    def _apply_style(self):
        pal = self.palette()
        bg = pal.color(QPalette.ColorRole.Window).darker(115)
        fg = pal.color(QPalette.ColorRole.WindowText)
        self.setAutoFillBackground(True)
        p = self.palette()
        p.setColor(QPalette.ColorRole.Window, bg)
        self.setPalette(p)
        cp = self._console.palette()
        cp.setColor(QPalette.ColorRole.Base, bg)
        cp.setColor(QPalette.ColorRole.Text, fg)
        self._console.setPalette(cp)

    def append(self, text):
        self._history.append(text)
        metrics = self._label.fontMetrics()
        elided = metrics.elidedText(text, Qt.TextElideMode.ElideRight, self._label.width() or 600)
        self._label.setText(elided)
        self._console.appendPlainText(text)
        sb = self._console.verticalScrollBar()
        sb.setValue(sb.maximum())
