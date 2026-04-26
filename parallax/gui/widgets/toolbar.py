from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QToolBar, QLabel, QLineEdit, QPushButton, QToolButton, QMenu,
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
    layers_changed = pyqtSignal(dict)
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

        self._layers_menu = QMenu(self)
        self._act_unverified = self._layers_menu.addAction("Unverified")
        self._act_unverified.setCheckable(True)
        self._act_unverified.setChecked(True)
        self._act_conf_high = self._layers_menu.addAction("  High")
        self._act_conf_high.setCheckable(True)
        self._act_conf_high.setChecked(True)
        self._act_conf_med = self._layers_menu.addAction("  Med")
        self._act_conf_med.setCheckable(True)
        self._act_conf_med.setChecked(True)
        self._act_conf_low = self._layers_menu.addAction("  Low")
        self._act_conf_low.setCheckable(True)
        self._act_conf_low.setChecked(True)
        self._layers_menu.addSeparator()
        self._act_known = self._layers_menu.addAction("Known")
        self._act_known.setCheckable(True)
        self._act_known.setChecked(False)
        self._act_bookmarked = self._layers_menu.addAction("Bookmarked")
        self._act_bookmarked.setCheckable(True)
        self._act_bookmarked.setChecked(True)
        self._act_viewed = self._layers_menu.addAction("Viewed")
        self._act_viewed.setCheckable(True)
        self._act_viewed.setChecked(False)

        self._cascading = False

        self._act_unverified.toggled.connect(self._on_unverified_toggled)
        for act in (self._act_conf_high, self._act_conf_med, self._act_conf_low):
            act.toggled.connect(self._on_tier_toggled)
        for act in (self._act_known, self._act_bookmarked, self._act_viewed):
            act.toggled.connect(self._on_layer_toggled)

        self._layers_btn = QToolButton()
        self._layers_btn.setText("Layers")
        self._layers_btn.setMenu(self._layers_menu)
        self._layers_btn.setPopupMode(
            QToolButton.ToolButtonPopupMode.InstantPopup)
        self._layers_btn.setFixedHeight(24)
        self._layers_btn.setStyleSheet(
            "QToolButton { padding: 0 8px; }"
            "QToolButton::menu-indicator { image: none; }"
        )
        self.addWidget(self._layers_btn)

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

    def get_layer_state(self):
        return {
            "unverified": self._act_unverified.isChecked(),
            "known": self._act_known.isChecked(),
            "bookmarked": self._act_bookmarked.isChecked(),
            "viewed": self._act_viewed.isChecked(),
            "conf_high": self._act_conf_high.isChecked(),
            "conf_med": self._act_conf_med.isChecked(),
            "conf_low": self._act_conf_low.isChecked(),
        }

    def _on_unverified_toggled(self, checked):
        if self._cascading:
            return
        self._cascading = True
        for act in (self._act_conf_high, self._act_conf_med, self._act_conf_low):
            act.setChecked(checked)
        self._cascading = False
        self._on_layer_toggled(checked)

    def _on_tier_toggled(self, _):
        if self._cascading:
            return
        self._cascading = True
        any_on = (self._act_conf_high.isChecked() or
                  self._act_conf_med.isChecked() or
                  self._act_conf_low.isChecked())
        self._act_unverified.setChecked(any_on)
        self._cascading = False
        self._on_layer_toggled(None)

    def _on_layer_toggled(self, _):
        self.layers_changed.emit(self.get_layer_state())

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
