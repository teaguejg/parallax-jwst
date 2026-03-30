from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QToolBar, QLabel, QLineEdit, QPushButton, QCheckBox,
)


class _TargetInput(QLineEdit):
    def keyPressEvent(self, event):
        if (event.modifiers() == (Qt.KeyboardModifier.ControlModifier |
                                   Qt.KeyboardModifier.ShiftModifier)
                and event.key() == Qt.Key.Key_V):
            self.paste()
            return
        super().keyPressEvent(event)


class ParallaxToolbar(QToolBar):
    run_requested = pyqtSignal(str)
    toggle_known = pyqtSignal(bool)
    settings_requested = pyqtSignal()
    search_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__("Main", parent)

        self.addWidget(QLabel("Target:"))

        self.target_input = _TargetInput()
        self.target_input.setPlaceholderText("e.g. NGC 3132 or 159.25 -58.633")
        self.target_input.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)
        self.target_input.returnPressed.connect(self._on_run_clicked)
        self.addWidget(self.target_input)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run_clicked)
        self.addWidget(self.run_button)

        self.known_checkbox = QCheckBox("Show known")
        self.known_checkbox.setChecked(False)
        self.known_checkbox.toggled.connect(self.toggle_known)
        self.addWidget(self.known_checkbox)

        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(self.settings_requested)
        self.addWidget(self.settings_button)

        self.addSeparator()

        self.addWidget(QLabel("Search:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Candidate ID")
        self.search_input.setFixedWidth(140)
        self.search_input.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.search_input.returnPressed.connect(self._on_search)
        self.addWidget(self.search_input)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self._on_search)
        self.addWidget(self.search_button)

        self.progress_label = QLabel()
        self.progress_label.setVisible(False)
        self.addWidget(self.progress_label)

    def _on_run_clicked(self):
        text = self.target_input.text().strip()
        if text:
            self.run_requested.emit(text)

    def set_progress(self, step, detail):
        self.progress_label.setText(f"{step}: {detail}")
        self.progress_label.setVisible(True)

    def clear_progress(self):
        self.progress_label.setText("")
        self.progress_label.setVisible(False)

    def _on_search(self):
        text = self.search_input.text().strip()
        if text:
            self.search_requested.emit(text)

    def set_running(self, running):
        self.run_button.setEnabled(not running)
        self.target_input.setEnabled(not running)
        if running:
            self.progress_label.setText("Running...")
            self.progress_label.setVisible(True)
