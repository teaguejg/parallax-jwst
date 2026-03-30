from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDockWidget, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

import os

import parallax as par
from parallax.config import config


def _spin_row(spinbox, default):
    row = QWidget()
    h = QHBoxLayout(row)
    h.setContentsMargins(0, 0, 0, 0)
    h.addWidget(spinbox, stretch=1)
    btn = QPushButton("Reset")
    btn.setFixedWidth(44)
    btn.clicked.connect(lambda: spinbox.setValue(default))
    h.addWidget(btn)
    return row


class SettingsPanel(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Settings", parent)
        self.setObjectName("Settings")
        self.setMinimumWidth(320)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        outer = QVBoxLayout(container)

        form = QFormLayout()

        path_row = QWidget()
        pr = QHBoxLayout(path_row)
        pr.setContentsMargins(0, 0, 0, 0)
        self._download_path = QLineEdit()
        pr.addWidget(self._download_path)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_download)
        pr.addWidget(browse)
        form.addRow("Download path", path_row)

        self._snr = QDoubleSpinBox()
        self._snr.setRange(0.5, 20.0)
        self._snr.setSingleStep(0.5)
        self._snr.setDecimals(1)
        form.addRow("SNR threshold", _spin_row(self._snr, 3.0))

        self._fwhm = QDoubleSpinBox()
        self._fwhm.setRange(0.5, 10.0)
        self._fwhm.setSingleStep(0.5)
        self._fwhm.setDecimals(1)
        form.addRow("Kernel FWHM", _spin_row(self._fwhm, 2.0))

        self._bg_box = QSpinBox()
        self._bg_box.setRange(4, 256)
        self._bg_box.setSingleStep(4)
        form.addRow("Background box size", _spin_row(self._bg_box, 50))

        self._bg_box2 = QSpinBox()
        self._bg_box2.setRange(0, 256)
        self._bg_box2.setSingleStep(4)
        form.addRow("Background box 2", _spin_row(self._bg_box2, 0))

        self._bg_interp = QComboBox()
        self._bg_interp.addItems(["zoom", "idw"])
        form.addRow("Background interp", self._bg_interp)

        self._min_px = QSpinBox()
        self._min_px.setRange(1, 200)
        self._min_px.setSingleStep(1)
        form.addRow("Min pixels", _spin_row(self._min_px, 25))

        self._radius = QDoubleSpinBox()
        self._radius.setRange(0.5, 30.0)
        self._radius.setSingleStep(0.5)
        self._radius.setDecimals(1)
        form.addRow("Search radius (arcsec)", _spin_row(self._radius, 2.0))

        self._ttl = QSpinBox()
        self._ttl.setRange(1, 365)
        self._ttl.setSingleStep(1)
        form.addRow("Catalog cache TTL (days)", _spin_row(self._ttl, 30))

        outer.addLayout(form)

        cache_row = QWidget()
        cr = QHBoxLayout(cache_row)
        cr.setContentsMargins(0, 4, 0, 4)
        clear_btn = QPushButton("Clear detection cache")
        clear_btn.setFixedWidth(160)
        clear_btn.clicked.connect(self._clear_cache)
        cr.addWidget(clear_btn)
        self._cache_label = QLabel("")
        self._cache_label.setStyleSheet("color: gray; margin-left: 6px;")
        cr.addWidget(self._cache_label)
        cr.addStretch()
        outer.addWidget(cache_row)

        btn_row = QWidget()
        bl = QHBoxLayout(btn_row)
        bl.setContentsMargins(0, 0, 0, 0)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._cancel)
        bl.addStretch()
        bl.addWidget(apply_btn)
        bl.addWidget(cancel_btn)
        outer.addWidget(btn_row)
        outer.addStretch()

        scroll.setWidget(container)
        self.setWidget(scroll)

    def showEvent(self, event):
        super().showEvent(event)
        self._cache_label.setText("")
        self._load_from_config()

    def _load_from_config(self):
        self._download_path.setText(
            os.path.normpath(config.get("data.download_path", ""))
        )
        self._snr.setValue(config.get("detection.snr_threshold", 3.0))
        self._fwhm.setValue(config.get("detection.kernel_fwhm", 2.0))
        self._bg_box.setValue(config.get("detection.background_box_size", 50))
        self._bg_box2.setValue(config.get("detection.background_box_size_2", 0))
        self._bg_interp.setCurrentText(config.get("detection.background_interp", "zoom"))
        self._min_px.setValue(config.get("detection.min_pixels", 25))
        self._radius.setValue(config.get("resolver.search_radius_arcsec", 2.0))
        self._ttl.setValue(config.get("cache.catalog_ttl_days", 30))

    def _browse_download(self):
        start = self._download_path.text() or ""
        d = QFileDialog.getExistingDirectory(
            self, "Select download folder", start
        )
        if d:
            self._download_path.setText(os.path.normpath(d))

    def _apply(self):
        config.set("data.download_path",
                   os.path.normpath(self._download_path.text()))
        config.set("detection.snr_threshold", self._snr.value())
        config.set("detection.kernel_fwhm", self._fwhm.value())
        config.set("detection.background_box_size", self._bg_box.value())
        config.set("detection.background_box_size_2", self._bg_box2.value())
        config.set("detection.background_interp", self._bg_interp.currentText())
        config.set("detection.min_pixels", self._min_px.value())
        config.set("resolver.search_radius_arcsec", self._radius.value())
        config.set("cache.catalog_ttl_days", self._ttl.value())
        try:
            config.save()
            QMessageBox.information(self, "Settings", "Settings saved.")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Save failed: {e}")

    def _clear_cache(self):
        result = par.survey.clear_cache()
        n = result.get("detection_entries_cleared", 0)
        if n > 0:
            self._cache_label.setText(f"Cleared {n} entries")
        else:
            self._cache_label.setText("Cache already empty")

    def _cancel(self):
        config.load()
        self._load_from_config()
        self.hide()
