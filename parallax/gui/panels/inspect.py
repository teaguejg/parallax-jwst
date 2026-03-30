import colorsys
import logging
import os
import warnings
from datetime import datetime, UTC

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QSplitter, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QMessageBox,
    QWidget,
)
from PyQt6.QtGui import QColor, QPainter, QLinearGradient, QPixmap

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval, make_lupton_rgb
import astropy.units as u

import parallax
from parallax import catalog

warnings.filterwarnings("ignore", category=FITSFixedWarning)

logger = logging.getLogger(__name__)

_FILTER_WL = {
    "F070W": 0.70, "F090W": 0.90, "F115W": 1.15, "F150W": 1.50,
    "F162M": 1.63, "F164N": 1.64, "F182M": 1.84, "F187N": 1.87,
    "F200W": 2.00, "F210M": 2.10, "F212N": 2.12, "F250M": 2.50,
    "F277W": 2.77, "F300M": 3.00, "F323N": 3.23, "F335M": 3.36,
    "F356W": 3.56, "F360M": 3.62, "F405N": 4.05, "F410M": 4.10,
    "F430M": 4.28, "F444W": 4.44, "F460M": 4.63, "F470N": 4.71,
    "F480M": 4.82,
}

_CLS_COLORS = {
    "unverified": "#c0392b",
    "known": "#a8d8ea",
}


def _sort_key(filt):
    wl = _FILTER_WL.get(filt.upper())
    if wl is not None:
        return (0, wl)
    return (1, filt)


def _find_sci_hdu(hdul):
    try:
        sci = hdul["SCI"]
        if sci.data is not None and sci.data.ndim == 2:
            return sci
    except KeyError:
        pass
    for hdu in hdul:
        if hdu.data is not None and hdu.data.ndim == 2 and hdu.data.size > 0:
            return hdu
    return None


def _border_median(data, width=3):
    mask = np.ones_like(data, dtype=bool)
    if data.shape[0] > 2*width and data.shape[1] > 2*width:
        mask[width:-width, width:-width] = False
    vals = data[mask]
    if np.all(np.isnan(vals)):
        return 0.0
    return float(np.nanmedian(vals))


def _palette_chromatic(sorted_filters):
    """STScI chromatic ordering: blue(short) to red(long) via HSV interpolation."""
    n = len(sorted_filters)
    if n == 0:
        return {}
    if n == 1:
        return {sorted_filters[0]: (1.0, 1.0, 1.0)}
    colors = {}
    for i, filt in enumerate(sorted_filters):
        hue = 0.67 * (1.0 - i / (n - 1))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        colors[filt] = (r, g, b)
    return colors


def _auto_colors(sorted_filters):
    return _palette_chromatic(sorted_filters)


def _is_narrowband(filt):
    return filt.upper().strip().endswith("N")


def _palette_hubble(sorted_filters):
    """Classic Hubble palette from HST ACS/WFC3 composite conventions."""
    blue = (0.18, 0.45, 0.90)
    green = (0.20, 0.85, 0.30)
    red = (0.90, 0.18, 0.08)
    n = len(sorted_filters)
    if n == 0:
        return {}
    if n == 1:
        return {sorted_filters[0]: (1.0, 1.0, 1.0)}
    if n == 2:
        return {sorted_filters[0]: blue, sorted_filters[1]: red}
    colors = {sorted_filters[0]: blue,
              sorted_filters[1]: green,
              sorted_filters[2]: red}
    for i in range(3, n):
        t = (i - 2) / (n - 2)
        r = green[0] + t * (red[0] - green[0])
        g = green[1] + t * (red[1] - green[1])
        b = green[2] + t * (red[2] - green[2])
        colors[sorted_filters[i]] = (r, g, b)
    return colors


def _palette_emission(sorted_filters):
    """Nebular emission: desaturated broadband, vivid narrowband."""
    n = len(sorted_filters)
    if n == 0:
        return {}
    if n == 1:
        return {sorted_filters[0]: (1.0, 1.0, 1.0)}
    colors = {}
    bb = [f for f in sorted_filters if not _is_narrowband(f)]
    bb_count = len(bb)
    bb_idx = 0
    for filt in sorted_filters:
        if _is_narrowband(filt):
            wl = _FILTER_WL.get(filt.upper(), 999.0)
            if wl < 2.5:
                colors[filt] = (0.0, 0.9, 0.85)
            else:
                colors[filt] = (0.85, 0.1, 0.75)
        else:
            if bb_count <= 1:
                hue = 0.33
            else:
                hue = 0.67 * (1.0 - bb_idx / (bb_count - 1))
            r, g, b = colorsys.hsv_to_rgb(hue, 0.5, 1.0)
            colors[filt] = (r, g, b)
            bb_idx += 1
    return colors


def _palette_h2(sorted_filters):
    n = len(sorted_filters)
    if n == 0:
        return {}
    if n == 1:
        return {sorted_filters[0]: (1.0, 1.0, 1.0)}
    nb_palette = [(0.55, 0.90, 0.05), (0.85, 0.95, 0.10), (0.95, 0.80, 0.05)]
    bb_palette = [(0.15, 0.25, 0.70), (0.20, 0.50, 0.75), (0.25, 0.65, 0.80)]
    colors = {}
    nb_idx = 0
    bb_idx = 0
    for filt in sorted_filters:
        if _is_narrowband(filt):
            colors[filt] = nb_palette[min(nb_idx, len(nb_palette) - 1)]
            nb_idx += 1
        else:
            colors[filt] = bb_palette[min(bb_idx, len(bb_palette) - 1)]
            bb_idx += 1
    return colors


def _palette_warm_dust(sorted_filters):
    n = len(sorted_filters)
    if n == 0:
        return {}
    if n == 1:
        return {sorted_filters[0]: (1.0, 0.6, 0.1)}
    colors = {}
    for i, filt in enumerate(sorted_filters):
        # hue from 0.58 (cool) to 0.08 (warm)
        hue = 0.58 - i * (0.50 / (n - 1))
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        colors[filt] = (r, g, b)
    return colors


def _palette_stellar(sorted_filters):
    n = len(sorted_filters)
    if n == 0:
        return {}
    short = (0.85, 0.90, 1.0)
    mid = (1.0, 1.0, 1.0)
    long = (1.0, 0.75, 0.15)
    if n == 1:
        return {sorted_filters[0]: short}
    colors = {}
    for i, filt in enumerate(sorted_filters):
        t = i / (n - 1)
        if t <= 0.5:
            s = t / 0.5
            rgb = tuple(a + s * (b - a) for a, b in zip(short, mid))
        else:
            s = (t - 0.5) / 0.5
            rgb = tuple(a + s * (b - a) for a, b in zip(mid, long))
        colors[filt] = rgb
    return colors


def _palette_infrared(sorted_filters):
    n = len(sorted_filters)
    if n == 0:
        return {}
    short = (0.10, 0.15, 0.70)
    mid = (0.75, 0.10, 0.65)
    long = (1.0, 0.95, 0.90)
    if n == 1:
        return {sorted_filters[0]: short}
    colors = {}
    for i, filt in enumerate(sorted_filters):
        t = i / (n - 1)
        if t <= 0.5:
            s = t / 0.5
            rgb = tuple(a + s * (b - a) for a, b in zip(short, mid))
        else:
            s = (t - 0.5) / 0.5
            rgb = tuple(a + s * (b - a) for a, b in zip(mid, long))
        colors[filt] = rgb
    return colors


_PALETTES = {
    "Chromatic": _palette_chromatic,
    "Hubble": _palette_hubble,
    "Emission": _palette_emission,
    "Molecular H2": _palette_h2,
    "Warm Dust": _palette_warm_dust,
    "Stellar": _palette_stellar,
    "Infrared": _palette_infrared,
}


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(
        int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    )


class _HueBar(QLabel):
    def __init__(self, width=220, height=18, parent=None):
        super().__init__(parent)
        self.setFixedSize(width, height)
        self._pm = QPixmap(width, height)
        self._render()
        self.on_hue_picked = None

    def _render(self):
        p = QPainter(self._pm)
        w = self._pm.width()
        h = self._pm.height()
        for x in range(w):
            hue = x / w
            c = QColor.fromHsvF(hue, 1.0, 1.0)
            p.setPen(c)
            p.drawLine(x, 0, x, h)
        p.end()
        self.setPixmap(self._pm)

    def mousePressEvent(self, ev):
        self._pick(ev)

    def mouseMoveEvent(self, ev):
        self._pick(ev)

    def _pick(self, ev):
        x = max(0, min(ev.position().x(), self.width() - 1))
        hue = x / self.width()
        if self.on_hue_picked:
            self.on_hue_picked(hue)


class _SVSquare(QLabel):
    def __init__(self, size=220, parent=None):
        super().__init__(parent)
        self._size = size
        self.setFixedSize(size, size)
        self._hue = 0.0
        self._sat = 1.0
        self._val = 1.0
        self._pm = QPixmap(size, size)
        self.on_sv_picked = None
        self._render()

    def set_hue(self, hue):
        self._hue = hue
        self._render()

    def set_sv(self, s, v):
        self._sat = s
        self._val = v
        self._draw_marker()

    def _render(self):
        sz = self._size
        p = QPainter(self._pm)
        for x in range(sz):
            s = x / sz
            for y in range(sz):
                v = 1.0 - y / sz
                p.setPen(QColor.fromHsvF(self._hue, s, v))
                p.drawPoint(x, y)
        p.end()
        self._draw_marker()

    def _draw_marker(self):
        pm = self._pm.copy()
        p = QPainter(pm)
        sz = self._size
        mx = int(self._sat * sz)
        my = int((1.0 - self._val) * sz)
        p.setPen(QColor(255, 255, 255))
        p.drawEllipse(mx - 5, my - 5, 10, 10)
        p.setPen(QColor(0, 0, 0))
        p.drawEllipse(mx - 4, my - 4, 8, 8)
        p.end()
        self.setPixmap(pm)

    def mousePressEvent(self, ev):
        self._pick(ev)

    def mouseMoveEvent(self, ev):
        self._pick(ev)

    def _pick(self, ev):
        sz = self._size
        x = max(0, min(ev.position().x(), sz - 1))
        y = max(0, min(ev.position().y(), sz - 1))
        self._sat = x / sz
        self._val = 1.0 - y / sz
        self._draw_marker()
        if self.on_sv_picked:
            self.on_sv_picked(self._sat, self._val)


class ColorPickerPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)

        self._hue = 0.0
        self._sat = 1.0
        self._val = 1.0
        self._current_filt = None
        self._on_change = None
        self._on_alpha_change = None
        self._updating = False

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 6, 10, 10)

        self._title = QLabel()
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title.setStyleSheet("font-weight: bold;")
        lay.addWidget(self._title)

        self._preview = QLabel()
        self._preview.setFixedSize(220, 80)
        lay.addWidget(self._preview, alignment=Qt.AlignmentFlag.AlignCenter)

        self._hue_bar = _HueBar(220, 18)
        self._hue_bar.on_hue_picked = self._on_hue_picked
        lay.addWidget(self._hue_bar, alignment=Qt.AlignmentFlag.AlignCenter)

        self._sv_square = _SVSquare(220)
        self._sv_square.on_sv_picked = self._on_sv_picked
        lay.addWidget(self._sv_square, alignment=Qt.AlignmentFlag.AlignCenter)

        form = QFormLayout()
        self._spin_r = QSpinBox(); self._spin_r.setRange(0, 255)
        self._spin_g = QSpinBox(); self._spin_g.setRange(0, 255)
        self._spin_b = QSpinBox(); self._spin_b.setRange(0, 255)
        self._spin_h = QSpinBox(); self._spin_h.setRange(0, 359)
        self._spin_s = QSpinBox(); self._spin_s.setRange(0, 100)
        self._spin_v = QSpinBox(); self._spin_v.setRange(0, 100)
        form.addRow("R", self._spin_r)
        form.addRow("G", self._spin_g)
        form.addRow("B", self._spin_b)
        form.addRow("H", self._spin_h)
        form.addRow("S", self._spin_s)
        form.addRow("V", self._spin_v)
        self._spin_a = QSpinBox(); self._spin_a.setRange(0, 100)
        form.addRow("A", self._spin_a)
        lay.addLayout(form)

        for sp in (self._spin_r, self._spin_g, self._spin_b):
            sp.valueChanged.connect(self._on_rgb_spin)
        for sp in (self._spin_h, self._spin_s, self._spin_v):
            sp.valueChanged.connect(self._on_hsv_spin)
        self._spin_a.valueChanged.connect(self._on_alpha_spin)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.hide)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

    def set_filter(self, filt, rgb):
        self._current_filt = filt
        self._title.setText(f"Color: {filt}")
        r, g, b = rgb
        self._hue, self._sat, self._val = colorsys.rgb_to_hsv(r, g, b)
        self._sync_all()

    def set_on_change(self, callback):
        self._on_change = callback

    def set_on_alpha_change(self, callback):
        self._on_alpha_change = callback

    def set_alpha(self, value):
        self._spin_a.blockSignals(True)
        self._spin_a.setValue(value)
        self._spin_a.blockSignals(False)

    def _emit(self):
        if self._on_change and self._current_filt:
            r, g, b = colorsys.hsv_to_rgb(self._hue, self._sat, self._val)
            self._on_change(self._current_filt, (r, g, b))

    def _on_hue_picked(self, hue):
        self._hue = hue
        self._sv_square.set_hue(hue)
        self._sync_all()
        self._emit()

    def _on_sv_picked(self, s, v):
        self._sat = s
        self._val = v
        self._sync_all()
        self._emit()

    def _on_rgb_spin(self):
        if self._updating:
            return
        r = self._spin_r.value() / 255.0
        g = self._spin_g.value() / 255.0
        b = self._spin_b.value() / 255.0
        self._hue, self._sat, self._val = colorsys.rgb_to_hsv(r, g, b)
        self._sync_all()
        self._emit()

    def _on_hsv_spin(self):
        if self._updating:
            return
        self._hue = self._spin_h.value() / 359.0
        self._sat = self._spin_s.value() / 100.0
        self._val = self._spin_v.value() / 100.0
        self._sync_all()
        self._emit()

    def _on_alpha_spin(self, val):
        if self._updating:
            return
        if self._on_alpha_change and self._current_filt:
            self._on_alpha_change(self._current_filt, val)

    def _sync_all(self):
        self._updating = True
        r, g, b = colorsys.hsv_to_rgb(self._hue, self._sat, self._val)

        self._spin_r.setValue(int(r * 255))
        self._spin_g.setValue(int(g * 255))
        self._spin_b.setValue(int(b * 255))
        self._spin_h.setValue(int(self._hue * 359))
        self._spin_s.setValue(int(self._sat * 100))
        self._spin_v.setValue(int(self._val * 100))

        hex_c = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255))
        self._preview.setStyleSheet(
            f"background-color: {hex_c}; border: 1px solid #666;")

        self._sv_square.set_hue(self._hue)
        self._sv_square.set_sv(self._sat, self._val)

        self._updating = False


class InspectWindow(QDialog):
    def __init__(self, candidate_id: str, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowTitle(f"Inspect: {candidate_id}")
        self.setMinimumSize(800, 600)

        self._candidate = catalog.get(candidate_id)
        if self._candidate is None:
            self._show_error("Candidate not found")
            return

        self._cutouts = {}
        self._sorted_filters = []
        self._det_snr = {}
        self._filter_colors = {}
        self._filter_enabled = {}
        self._color_buttons = {}
        self._filter_checkboxes = {}
        self._filter_alphas = {}
        self._active_picker_filt = None
        self._custom_colors = {}

        self._outer = QHBoxLayout(self)

        left_widget = QWidget()
        self._root = QVBoxLayout(left_widget)
        self._root.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        hdr.addWidget(QLabel(f"<b>{self._candidate.id}</b>"))
        hdr.addWidget(QLabel(f"RA {self._candidate.ra:.5f}"))
        hdr.addWidget(QLabel(f"Dec {self._candidate.dec:.5f}"))
        hdr.addWidget(QLabel(f"SNR {self._candidate.snr:.1f}"))

        cls_label = QLabel(self._candidate.classification)
        color = _CLS_COLORS.get(self._candidate.classification, "#888")
        cls_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        hdr.addWidget(cls_label)

        if self._candidate.catalog_matches:
            parts = []
            for m in self._candidate.catalog_matches:
                tag = m.object_type or m.source_id
                parts.append(f"{m.catalog}: {tag}")
            hdr.addWidget(QLabel(", ".join(parts)))

        hdr.addStretch()
        self._root.addLayout(hdr)

        # controls row placeholder -- added after _extract_cutouts if data available
        self._controls_row = None

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._root.addWidget(self._splitter, stretch=1)

        self._composite_fig = Figure(figsize=(5, 5))
        self._composite_canvas = FigureCanvasQTAgg(self._composite_fig)
        self._strip_fig = Figure(figsize=(3, 5))
        self._strip_canvas = FigureCanvasQTAgg(self._strip_fig)
        self._splitter.addWidget(self._composite_canvas)
        self._splitter.addWidget(self._strip_canvas)
        self._splitter.setStretchFactor(0, 2)
        self._splitter.setStretchFactor(1, 1)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_inspection)
        btn_row.addWidget(save_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_row.addWidget(close_btn)
        self._root.addLayout(btn_row)

        self._outer.addWidget(left_widget, stretch=1)

        self._color_picker = ColorPickerPanel()
        self._color_picker.set_on_change(self._on_picker_change)
        self._color_picker.set_on_alpha_change(self._on_picker_alpha_change)
        self._color_picker.hide()
        self._outer.addWidget(self._color_picker, stretch=0)

        try:
            self._build_figures()
        except Exception:
            logger.exception("failed to build inspect figures")
            self._splitter.hide()
            if self._controls_row is not None:
                # hide controls widget if it was added
                for i in range(self._root.count()):
                    item = self._root.itemAt(i)
                    if item and item.layout() == self._controls_row:
                        break
            err = QLabel("Inspection failed, check log")
            err.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._root.insertWidget(1, err)

    def closeEvent(self, event):
        # stop matplotlib canvas timers before Qt tears down the widget tree
        for attr in ("_composite_canvas", "_strip_canvas"):
            canvas = getattr(self, attr, None)
            if canvas is None:
                continue
            try:
                # kill any pending draw timer
                if hasattr(canvas, "_timer"):
                    canvas._timer.stop()
                canvas.close()
            except Exception:
                pass
        for attr in ("_composite_fig", "_strip_fig"):
            fig = getattr(self, attr, None)
            if fig is None:
                continue
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except Exception:
                pass
        self.deleteLater()
        super().closeEvent(event)

    def _show_error(self, msg):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(msg))

    def _build_figures(self):
        self._extract_cutouts()
        if not self._cutouts:
            return

        # filter count in header
        n = len(self._sorted_filters)
        filter_names = ", ".join(self._sorted_filters)
        filter_label = QLabel(f"Filters: {n} ({filter_names})")
        filter_label.setStyleSheet("color: #888; font-size: 10px;")
        # find the header layout (index 0) and append
        hdr_item = self._root.itemAt(0)
        if hdr_item and hdr_item.layout():
            hdr_item.layout().addWidget(filter_label)

        self._add_controls()

        if n < 3:
            note = QLabel(
                "Color mapping is approximate, fewer than 3 filters detected."
            )
            note.setStyleSheet("color: #888; font-size: 10px;")
            # insert after header (0), ctrl row (1), ctrl2 row (2)
            insert_idx = 3 if self._controls_row else 1
            self._root.insertWidget(insert_idx, note)

        self._rebuild_composite()

    def _extract_cutouts(self):
        from parallax import archive
        from parallax.config import config

        fits_by_filter = archive.get_fits_per_filter(self._candidate.id)

        if not fits_by_filter:
            self._cutouts = {}
            ax = self._composite_fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "No source files found for this candidate.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            ax.set_axis_off()
            self._composite_canvas.draw()
            self._strip_canvas.hide()
            return

        size = config.get("detection.cutout_size", 80)
        coord = SkyCoord(self._candidate.ra, self._candidate.dec, unit="deg")

        cutouts = {}
        for filt, path in fits_by_filter.items():
            try:
                with fits.open(path) as hdul:
                    sci = _find_sci_hdu(hdul)
                    if sci is None:
                        continue
                    wcs = WCS(sci.header)
                    data = sci.data.astype(np.float64)
                    cut = Cutout2D(data, coord, size * u.pixel, wcs=wcs,
                                    mode='partial', fill_value=np.nan)
                    sub = cut.data - _border_median(cut.data)
                    if not np.all(np.isnan(sub)):
                        cutouts[filt] = sub
            except Exception:
                continue

        if not cutouts:
            self._cutouts = {}
            ax = self._composite_fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "Could not extract cutout (WCS mismatch or edge detection).",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            ax.set_axis_off()
            self._composite_canvas.draw()
            self._strip_canvas.hide()
            return

        self._cutouts = cutouts
        self._sorted_filters = sorted(cutouts.keys(), key=_sort_key)

        self._det_snr = {}
        for d in self._candidate.detections:
            self._det_snr[d.filter] = d.snr

    def _add_controls(self):
        ctrl = QHBoxLayout()
        self._controls_row = ctrl

        self._filter_colors = _palette_chromatic(self._sorted_filters)
        self._color_buttons = {}
        self._filter_checkboxes = {}

        for filt in self._sorted_filters:
            self._filter_enabled[filt] = filt in self._det_snr
            default_alpha = 100
            self._filter_alphas[filt] = default_alpha
            cb = QCheckBox()
            cb.setChecked(self._filter_enabled[filt])
            cb.setFixedWidth(18)
            cb.stateChanged.connect(lambda state, f=filt: self._on_filter_toggle(f))
            ctrl.addWidget(cb)
            btn = QPushButton(filt)
            btn.setFixedHeight(26)
            rgb = self._filter_colors[filt]
            self._apply_btn_style(btn, rgb)
            btn.clicked.connect(lambda checked, f=filt: self._open_color_picker(f))
            ctrl.addWidget(btn)
            self._color_buttons[filt] = btn
            self._filter_checkboxes[filt] = cb

        ctrl.addStretch()

        ctrl2 = QHBoxLayout()
        self._controls_row2 = ctrl2

        ctrl2.addWidget(QLabel("Stretch:"))
        self._stretch_spin = QDoubleSpinBox()
        self._stretch_spin.setRange(0.1, 5.0)
        self._stretch_spin.setSingleStep(0.1)
        self._stretch_spin.setDecimals(1)
        self._stretch_spin.setValue(0.5)
        self._stretch_spin.setToolTip("Stretch")
        self._stretch_spin.valueChanged.connect(self._rebuild_composite)
        ctrl2.addWidget(self._stretch_spin)

        ctrl2.addWidget(QLabel("Q:"))
        self._q_spin = QDoubleSpinBox()
        self._q_spin.setRange(1.0, 20.0)
        self._q_spin.setSingleStep(1.0)
        self._q_spin.setDecimals(0)
        self._q_spin.setValue(10.0)
        self._q_spin.setToolTip("Q")
        self._q_spin.valueChanged.connect(self._rebuild_composite)
        ctrl2.addWidget(self._q_spin)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._reset_controls)
        ctrl2.addWidget(reset_btn)

        self._palette_combo = QComboBox()
        for name in _PALETTES:
            self._palette_combo.addItem(name)
        self._palette_combo.addItem("Custom")
        self._palette_combo.setFixedWidth(110)
        self._palette_combo.currentTextChanged.connect(self._on_palette_changed)
        self._palette_combo.blockSignals(True)
        self._palette_combo.setCurrentText("Chromatic")
        self._palette_combo.blockSignals(False)
        ctrl2.addWidget(self._palette_combo)

        ctrl2.addStretch()

        # insert between header (index 0) and splitter (index 1)
        self._root.insertLayout(1, ctrl)
        self._root.insertLayout(2, ctrl2)

    def _apply_btn_style(self, btn, rgb, active=False):
        hex_c = _rgb_to_hex(rgb)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt = "#000" if lum > 0.5 else "#fff"
        border = "2px solid #fff" if active else "1px solid #666"
        btn.setStyleSheet(
            f"background-color: {hex_c}; color: {txt}; "
            f"border: {border}; padding: 2px 6px;"
        )

    def _update_btn_highlight(self, active_filt):
        for filt, btn in self._color_buttons.items():
            self._apply_btn_style(btn, self._filter_colors[filt], active=(filt == active_filt))

    def _on_filter_toggle(self, filt):
        self._filter_enabled[filt] = self._filter_checkboxes[filt].isChecked()
        self._rebuild_composite()

    def _on_picker_alpha_change(self, filt, val):
        self._filter_alphas[filt] = val
        self._rebuild_composite()

    def _open_color_picker(self, filt):
        if not self._filter_enabled.get(filt, True):
            return
        if self._color_picker.isVisible() and self._active_picker_filt == filt:
            # toggle off
            self.resize(self.width() - self._color_picker.width(), self.height())
            self._color_picker.hide()
            self._active_picker_filt = None
            self._update_btn_highlight(None)
            return
        self._active_picker_filt = filt
        self._color_picker.set_filter(filt, self._filter_colors[filt])
        self._color_picker.set_alpha(self._filter_alphas.get(filt, 100))
        if not self._color_picker.isVisible():
            self.resize(self.width() + self._color_picker.width(), self.height())
        self._color_picker.show()
        self._update_btn_highlight(filt)

    def _on_picker_change(self, filt, rgb):
        self._filter_colors[filt] = rgb
        self._apply_btn_style(self._color_buttons[filt], rgb)
        if self._palette_combo.currentText() != "Custom":
            self._palette_combo.blockSignals(True)
            self._palette_combo.setCurrentText("Custom")
            self._palette_combo.blockSignals(False)
        self._rebuild_composite()

    def _on_palette_changed(self, name):
        if name == "Custom":
            # restore saved custom colors if available
            if self._custom_colors:
                for filt in self._sorted_filters:
                    if filt in self._custom_colors:
                        self._filter_colors[filt] = self._custom_colors[filt]
                for filt, btn in self._color_buttons.items():
                    self._apply_btn_style(btn, self._filter_colors[filt],
                                          active=(filt == self._active_picker_filt))
                if self._color_picker.isVisible():
                    f = self._color_picker._current_filt
                    if f and f in self._filter_colors:
                        self._color_picker.set_filter(f, self._filter_colors[f])
                self._rebuild_composite()
            return
        # save current colors as custom before overwriting
        self._custom_colors = dict(self._filter_colors)
        fn = _PALETTES.get(name)
        if not fn:
            return
        new_colors = fn(self._sorted_filters)
        for filt in self._sorted_filters:
            if self._filter_enabled.get(filt, True):
                self._filter_colors[filt] = new_colors[filt]
        for filt, btn in self._color_buttons.items():
            self._apply_btn_style(btn, self._filter_colors[filt],
                                  active=(filt == self._active_picker_filt))
        for f in self._sorted_filters:
            self._filter_alphas[f] = 100
        if self._color_picker.isVisible():
            f = self._color_picker._current_filt
            if f and f in self._filter_colors:
                self._color_picker.set_filter(f, self._filter_colors[f])
            if f:
                self._color_picker.set_alpha(100)
        self._rebuild_composite()

    def _reset_controls(self):
        self._filter_colors = _palette_chromatic(self._sorted_filters)
        for filt in self._sorted_filters:
            self._filter_enabled[filt] = filt in self._det_snr
            cb = self._filter_checkboxes[filt]
            cb.blockSignals(True)
            cb.setChecked(self._filter_enabled[filt])
            cb.blockSignals(False)
            self._filter_alphas[filt] = 100
        for filt, btn in self._color_buttons.items():
            self._apply_btn_style(btn, self._filter_colors[filt],
                                  active=(filt == self._active_picker_filt))
        self._stretch_spin.setValue(0.5)
        self._q_spin.setValue(10.0)
        self._palette_combo.blockSignals(True)
        self._palette_combo.setCurrentText("Chromatic")
        self._palette_combo.blockSignals(False)
        if self._color_picker.isVisible():
            f = self._color_picker._current_filt
            if f and f in self._filter_colors:
                self._color_picker.set_filter(f, self._filter_colors[f])
                self._color_picker.set_alpha(self._filter_alphas.get(f, 100))
        self._rebuild_composite()

    def _rebuild_composite(self):
        self._composite_fig.clear()
        self._strip_fig.clear()

        stretch_val = self._stretch_spin.value()
        q_val = self._q_spin.value()

        ref = next(iter(self._cutouts.values()))
        shape = ref.shape

        active = [f for f in self._sorted_filters
                  if self._filter_enabled.get(f, True)]

        # normalize cutout shapes -- edge cutouts may be off by a pixel
        if active:
            min_h = min(self._cutouts[f].shape[0] for f in active)
            min_w = min(self._cutouts[f].shape[1] for f in active)
            active_cutouts = {f: self._cutouts[f][:min_h, :min_w] for f in active}
            shape = (min_h, min_w)

        interval = ZScaleInterval()
        stretch = AsinhStretch()

        if not active:
            ax = self._composite_fig.add_subplot(111)
            ax.text(0.5, 0.5, "All filters disabled",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#888888")
            ax.set_axis_off()
            self._composite_fig.tight_layout()
            self._composite_canvas.draw()
        else:
            r_plane = np.zeros(shape, dtype=np.float64)
            g_plane = np.zeros(shape, dtype=np.float64)
            b_plane = np.zeros(shape, dtype=np.float64)

            detected = {f: self._det_snr[f] for f in active if f in self._det_snr}
            if detected:
                max_snr = max(detected.values())
                weights = {}
                for f in active:
                    if f in detected:
                        weights[f] = min(1.0, detected[f] / max_snr)
                    else:
                        # manually enabled but not detected - half weight
                        weights[f] = 0.5
            else:
                weights = {f: 1.0 for f in active}

            for filt in active:
                data = active_cutouts[filt]
                norm = ImageNormalize(data, interval=interval, stretch=stretch)
                normed = norm(data)
                normed = np.nan_to_num(np.asarray(normed), nan=0.0)
                w = weights[filt] * (self._filter_alphas.get(filt, 100) / 100.0)
                cr, cg, cb = self._filter_colors.get(filt, (1.0, 1.0, 1.0))
                r_plane += normed * cr * w
                g_plane += normed * cg * w
                b_plane += normed * cb * w

            r_plane = np.clip(r_plane, 0, 1)
            g_plane = np.clip(g_plane, 0, 1)
            b_plane = np.clip(b_plane, 0, 1)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rgb = make_lupton_rgb(r_plane, g_plane, b_plane,
                                      stretch=stretch_val, Q=q_val)

            ax = self._composite_fig.add_subplot(111)
            ax.imshow(rgb, origin="lower")

            pairs = []
            for f in active:
                s = f"{f}:{_rgb_to_hex(self._filter_colors[f])}"
                a = self._filter_alphas.get(f, 100)
                if a != 100:
                    s += f" a={a}"
                pairs.append(s)
            ax.set_title("  ".join(pairs), fontsize=7)
            ax.set_axis_off()

            self._composite_fig.tight_layout()
            self._composite_canvas.draw()

        # filter strip -- all filters, dim disabled ones
        n_filters = len(self._sorted_filters)
        for i, filt in enumerate(self._sorted_filters):
            ax = self._strip_fig.add_subplot(n_filters, 1, i + 1)
            d = self._cutouts[filt]
            norm = ImageNormalize(d, interval=ZScaleInterval(), stretch=AsinhStretch())
            enabled = self._filter_enabled.get(filt, True)
            ax.imshow(d, origin="lower", cmap="gray", norm=norm,
                      alpha=1.0 if enabled else 0.35)
            snr_val = self._det_snr.get(filt)
            title = f"{filt}" + (f"  SNR {snr_val:.1f}" if snr_val else "  (no detection)")
            ax.set_title(title, fontsize=8,
                         color="black" if enabled else "#888888")
            ax.set_axis_off()

        self._strip_fig.tight_layout()
        self._strip_canvas.draw()

    def _save_inspection(self):
        try:
            from parallax.config import config
            from parallax.types import _target_slug
            from parallax import archive as _arc

            rpt = _arc.get_report(self._candidate.report_id)
            target = rpt.target if rpt else "unknown"
            slug = _target_slug(target)
            reports_path = config.get("data.reports_path", "data/reports")
            save_dir = os.path.join(reports_path, slug, "inspections",
                                    self._candidate.id)
            os.makedirs(save_dir, exist_ok=True)

            cid = self._candidate.id
            composite_path = os.path.join(save_dir, f"{cid}_composite.png")
            self._composite_fig.savefig(composite_path, dpi=150, bbox_inches="tight")
            self._strip_fig.savefig(
                os.path.join(save_dir, f"{cid}_strip.png"), dpi=150, bbox_inches="tight"
            )

            lines = [
                f"# Inspection: {self._candidate.id}",
                f"{datetime.now(UTC).isoformat()}",
                "",
                "## Candidate",
                f"- ID: {self._candidate.id}",
                f"- RA: {self._candidate.ra:.6f}",
                f"- Dec: {self._candidate.dec:.6f}",
                f"- SNR: {self._candidate.snr:.2f}",
                f"- Classification: {self._candidate.classification}",
                f"- Report: {self._candidate.report_id}",
                f"- Parallax: {parallax.__version__}",
                "",
                "## Detections",
                "| Filter | SNR | Flux |",
                "|--------|-----|------|",
            ]
            dets = sorted(self._candidate.detections, key=lambda d: d.snr, reverse=True)
            for d in dets:
                lines.append(f"| {d.filter} | {d.snr:.2f} | {d.flux:.2f} |")

            lines.append("")
            lines.append("## Catalog Matches")
            lines.append('| Catalog | ID | Sep(") | Type |')
            lines.append("|---------|-----|--------|------|")
            for m in self._candidate.catalog_matches:
                lines.append(f"| {m.catalog} | {m.source_id} | {m.separation_arcsec:.2f} | {m.object_type or ''} |")

            stretch = self._stretch_spin.value() if hasattr(self, '_stretch_spin') else 0.5
            q = self._q_spin.value() if hasattr(self, '_q_spin') else 10.0

            lines.extend(["", "## Color Mapping"])
            for filt in self._sorted_filters:
                rgb = self._filter_colors.get(filt, (1, 1, 1))
                lines.append(f"- {filt}: {_rgb_to_hex(rgb)}")
            lines.append(f"Stretch: {stretch}  Q: {q}")

            with open(os.path.join(save_dir, f"{cid}_summary.md"), "w") as f:
                f.write("\n".join(lines) + "\n")

            from parallax.gui.platform import reveal_file
            reveal_file(composite_path)

        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
