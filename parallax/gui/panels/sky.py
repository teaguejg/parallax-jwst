import colorsys
import logging
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import pyqtSignal, Qt, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QStackedWidget, QLabel,
    QProgressBar, QPushButton,
)

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval

warnings.filterwarnings("ignore", category=FITSFixedWarning)

logger = logging.getLogger(__name__)

_COLORS = {
    "unverified": "#c0392b",
    "known": "#4a90d9",
}


def _conf_color(conf):
    if conf >= 0.75:
        return "#c0392b"
    elif conf >= 0.50:
        return "#e67e22"
    return "#7f8c8d"

_STEP_ORDER = {
    "acquire": 1, "downloading": 2, "detect": 3,
    "merge": 4, "resolve": 5, "report": 6,
    "chart": 7, "cutout": 7,
}

_FILTER_WL = {
    "F070W": 0.70, "F090W": 0.90, "F115W": 1.15, "F150W": 1.50,
    "F162M": 1.63, "F164N": 1.64, "F182M": 1.84, "F187N": 1.87,
    "F200W": 2.00, "F210M": 2.10, "F212N": 2.12, "F250M": 2.50,
    "F277W": 2.77, "F300M": 3.00, "F323N": 3.23, "F335M": 3.36,
    "F356W": 3.56, "F360M": 3.62, "F405N": 4.05, "F410M": 4.10,
    "F430M": 4.28, "F444W": 4.44, "F460M": 4.63, "F470N": 4.71,
    "F480M": 4.82,
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


def _auto_colors(sorted_filters):
    n = len(sorted_filters)
    if n == 0:
        return {}
    if n == 1:
        return {sorted_filters[0]: (1.0, 1.0, 1.0)}
    if n == 2:
        return {sorted_filters[0]: (0.0, 0.0, 1.0),
                sorted_filters[1]: (1.0, 0.0, 0.0)}
    colors = {}
    for i, filt in enumerate(sorted_filters):
        hue = 0.67 * (1.0 - i / (n - 1))
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors[filt] = (r, g, b)
    return colors


def _downsample(data, max_dim=1024):
    """Bin-downsample to keep longest axis <= max_dim."""
    h, w = data.shape
    factor = max(1, max(h, w) // max_dim)
    if factor == 1:
        return data
    nh = (h // factor) * factor
    nw = (w // factor) * factor
    trimmed = data[:nh, :nw]
    return trimmed.reshape(nh // factor, factor, nw // factor, factor).mean(axis=(1, 3))


class CompositeWorker(QThread):
    composite_ready = pyqtSignal(np.ndarray, object)
    composite_failed = pyqtSignal()

    def __init__(self, report_id, parent=None):
        super().__init__(parent)
        self._report_id = report_id

    def run(self):
        try:
            from parallax import archive
            fits_map = archive.get_fits_for_report(self._report_id)
            if not fits_map:
                self.composite_failed.emit()
                return

            arrays = {}
            wcs_out = None
            for filt, path in fits_map.items():
                try:
                    with fits.open(path) as hdul:
                        sci = _find_sci_hdu(hdul)
                        if sci is None:
                            continue
                        if wcs_out is None:
                            try:
                                wcs_out = WCS(sci.header)
                            except Exception:
                                pass
                        arrays[filt] = sci.data.astype(np.float64)
                except Exception:
                    continue

            if not arrays:
                self.composite_failed.emit()
                return

            sorted_f = sorted(arrays.keys(), key=_sort_key)
            colors = _auto_colors(sorted_f)

            # downsample all to same shape based on first
            ds = {}
            for filt in sorted_f:
                ds[filt] = _downsample(arrays[filt])

            # find common shape (use smallest dims)
            min_h = min(a.shape[0] for a in ds.values())
            min_w = min(a.shape[1] for a in ds.values())

            interval = ZScaleInterval()
            stretch = AsinhStretch()

            r_plane = np.zeros((min_h, min_w), dtype=np.float64)
            g_plane = np.zeros((min_h, min_w), dtype=np.float64)
            b_plane = np.zeros((min_h, min_w), dtype=np.float64)

            for filt in sorted_f:
                arr = ds[filt][:min_h, :min_w]
                norm = ImageNormalize(arr, interval=interval, stretch=stretch)
                normed = np.nan_to_num(norm(arr), nan=0.0)
                cr, cg, cb = colors[filt]
                r_plane += normed * cr
                g_plane += normed * cg
                b_plane += normed * cb

            rgb = np.clip(np.stack([r_plane, g_plane, b_plane], axis=-1), 0, 1)
            self.composite_ready.emit(rgb, wcs_out)

        except Exception:
            logger.exception("composite worker failed")
            self.composite_failed.emit()


_MODE_SCATTER = 0
_MODE_COMPOSITE = 1
_MODE_BOTH = 2


class SkyPanel(QWidget):
    candidate_selected = pyqtSignal(str)
    candidate_deselected = pyqtSignal()
    candidate_inspected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        self._progress_page = QWidget()
        pg_layout = QVBoxLayout(self._progress_page)
        pg_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedWidth(300)
        self._step_label = QLabel()
        self._step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._step_label.setStyleSheet("color: #888; font-size: 11px;")
        pg_layout.addWidget(self._status_label)
        pg_layout.addWidget(self._progress_bar)
        pg_layout.addWidget(self._step_label)
        self._stack.addWidget(self._progress_page)

        self._plot_page = QWidget()
        plot_layout = QVBoxLayout(self._plot_page)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self._zoom_row = QHBoxLayout()
        self._zoom_row_widget = QWidget()
        zr_inner = QHBoxLayout(self._zoom_row_widget)
        zr_inner.setContentsMargins(4, 4, 0, 0)
        self._zoom_btns = []
        for label, slot_name in [("+", "_zoom_in"), ("-", "_zoom_out"), ("Reset", "_zoom_reset")]:
            btn = QPushButton(label)
            btn.setFixedWidth(40 if label == "Reset" else 26)
            btn.setFixedHeight(22)
            btn.setStyleSheet("font-size: 11px; padding: 0;")
            btn.clicked.connect(getattr(self, slot_name))
            zr_inner.addWidget(btn)
            self._zoom_btns.append(btn)
        zr_inner.addStretch()

        # field mode button
        self._field_btn = QPushButton("Field")
        self._field_btn.setFixedWidth(52)
        self._field_btn.setFixedHeight(22)
        self._field_btn.setStyleSheet("font-size: 11px; padding: 0;")
        self._field_btn.setEnabled(False)
        self._field_btn.clicked.connect(self._cycle_mode)
        zr_inner.addWidget(self._field_btn)

        plot_layout.addWidget(self._zoom_row_widget)

        self._fig = Figure(figsize=(8, 8))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        plot_layout.addWidget(self._canvas)
        self._stack.addWidget(self._plot_page)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)

        self._composite_page = QWidget()
        comp_layout = QVBoxLayout(self._composite_page)
        comp_layout.setContentsMargins(0, 0, 0, 0)

        self._comp_toolbar = QWidget()
        ct_layout = QHBoxLayout(self._comp_toolbar)
        ct_layout.setContentsMargins(4, 4, 0, 0)
        ct_layout.addStretch()
        self._comp_field_btn = QPushButton("Scatter")
        self._comp_field_btn.setFixedWidth(52)
        self._comp_field_btn.setFixedHeight(22)
        self._comp_field_btn.setStyleSheet("font-size: 11px; padding: 0;")
        self._comp_field_btn.clicked.connect(self._cycle_mode)
        ct_layout.addWidget(self._comp_field_btn)
        comp_layout.addWidget(self._comp_toolbar)

        self._comp_fig = Figure(figsize=(8, 8))
        self._comp_ax = self._comp_fig.add_subplot(111)
        self._comp_canvas = FigureCanvasQTAgg(self._comp_fig)
        comp_layout.addWidget(self._comp_canvas)
        self._stack.addWidget(self._composite_page)

        self._both_page = QWidget()
        both_layout = QVBoxLayout(self._both_page)
        both_layout.setContentsMargins(0, 0, 0, 0)

        self._both_toolbar = QWidget()
        bt_layout = QHBoxLayout(self._both_toolbar)
        bt_layout.setContentsMargins(4, 4, 0, 0)
        bt_layout.addStretch()
        self._both_field_btn = QPushButton("Both")
        self._both_field_btn.setFixedWidth(52)
        self._both_field_btn.setFixedHeight(22)
        self._both_field_btn.setStyleSheet("font-size: 11px; padding: 0;")
        self._both_field_btn.clicked.connect(self._cycle_mode)
        bt_layout.addWidget(self._both_field_btn)
        both_layout.addWidget(self._both_toolbar)

        self._both_container = QHBoxLayout()
        self._both_left = QLabel()
        self._both_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._both_right = QLabel()
        self._both_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._both_container.addWidget(self._both_left)
        self._both_container.addWidget(self._both_right)
        both_layout.addLayout(self._both_container, stretch=1)
        self._stack.addWidget(self._both_page)

        self._candidates = []
        self._scatter_known = None
        self._scatter_other = {}
        self._selected_marker = None
        self._current_report_id = None
        self._original_xlim = None
        self._original_ylim = None
        self._selected_candidate = None

        self._composite_rgb = None
        self._composite_wcs = None
        self._composite_report_id = None
        self._composite_worker = None
        self._view_mode = _MODE_SCATTER
        self._known_visible = False

        self.show_idle()

    def show_idle(self):
        self._current_report_id = None
        self._status_label.setText("")
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._status_label.setVisible(False)
        self._step_label.setVisible(False)
        self._stack.setCurrentIndex(0)

    def show_progress(self, step, detail, value):
        self._status_label.setText(f"{step.capitalize()}: {detail}")
        self._status_label.setVisible(True)

        # indeterminate bar for acquire/downloading, determinate for others
        if step in ("acquire", "downloading"):
            self._progress_bar.setRange(0, 0)
        else:
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(value)

        self._progress_bar.setVisible(True)

        pos = _STEP_ORDER.get(step)
        if pos is not None:
            self._step_label.setText(f"Step {pos} of 7")
            self._step_label.setVisible(True)
        else:
            self._step_label.setVisible(False)

        self._stack.setCurrentIndex(0)

    def show_plot(self):
        self._view_mode = _MODE_SCATTER
        self._update_mode_ui()

    def load_report(self, report):
        self._ax.clear()
        self._current_report_id = report.id
        self._candidates = list(report.candidates) if report.candidates else []
        self._scatter_known = None
        self._scatter_other = {}
        self._scatter_bookmarked = None
        self._selected_marker = None
        self._view_mode = _MODE_SCATTER

        for cls, color in _COLORS.items():
            subset = [c for c in self._candidates if c.classification == cls]
            if not subset:
                continue
            ras = [c.ra for c in subset]
            decs = [c.dec for c in subset]
            snrs = np.array([c.snr for c in subset])
            if cls == "known":
                sizes = np.clip(snrs * 10, 20, 40)
                sc = self._ax.scatter(
                    ras, decs, s=sizes, c=color,
                    label=cls, alpha=0.6, edgecolors="none",
                    linewidth=0,
                )
                self._scatter_known = sc
                sc.set_visible(False)
                sc.set_label("known (hidden)")
            else:
                sizes = np.clip(snrs * 10, 20, 200)
                colors = [_conf_color(c.confidence) for c in subset]
                sc = self._ax.scatter(
                    ras, decs, s=sizes, c=colors,
                    label="_nolegend_", alpha=0.8, edgecolors="k", linewidth=0.5,
                )
                self._scatter_other[cls] = sc

        # legend entries for confidence tiers
        for label, lc in [
            ("unverified (high)", "#c0392b"),
            ("unverified (med)", "#e67e22"),
            ("unverified (low)", "#7f8c8d"),
        ]:
            self._ax.scatter([], [], s=40, c=lc,
                             label=label, edgecolors="k", linewidth=0.5)

        bm = [c for c in self._candidates if "bookmarked" in (c.tags or [])]
        if bm:
            bm_ras = [c.ra for c in bm]
            bm_decs = [c.dec for c in bm]
            bm_snrs = np.array([c.snr for c in bm])
            bm_sizes = np.clip(bm_snrs * 12, 30, 120)
            self._scatter_bookmarked = self._ax.scatter(
                bm_ras, bm_decs, s=bm_sizes, c="#f1c40f",
                label="bookmarked", alpha=0.9, edgecolors="#c8a000",
                linewidth=1.0, zorder=5,
            )

        self._ax.invert_xaxis()
        self._ax.set_xlabel("RA (deg)")
        self._ax.set_ylabel("Dec (deg)")

        # title is target + run date; filters go in corner annotation
        date_str = report.created_at.strftime("%Y-%m-%d") if report.created_at else ""
        self._ax.set_title(f"{report.target}  {date_str}")
        if report.filters:
            filters_str = " ".join(sorted(set(report.filters)))
            self._ax.annotate(
                filters_str, xy=(0.99, 0.01), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=7, color="#555555",
            )

        self._ax.legend(fontsize=8, loc="upper right")
        self._ax.grid(True, alpha=0.3)
        self._original_xlim = self._ax.get_xlim()
        self._original_ylim = self._ax.get_ylim()
        self._fig.tight_layout()
        self._canvas.draw()

        if self._composite_report_id != report.id:
            self._composite_rgb = None
            self._composite_wcs = None
            self._composite_report_id = None
            self._field_btn.setEnabled(False)
            self._start_composite_worker(report.id)
        else:
            self._field_btn.setEnabled(self._composite_rgb is not None)

        self._update_mode_ui()

    def _start_composite_worker(self, report_id):
        if self._composite_worker is not None and self._composite_worker.isRunning():
            return
        self._composite_worker = CompositeWorker(report_id, self)
        self._composite_worker.composite_ready.connect(
            lambda rgb, wcs, rid=report_id: self._on_composite_ready(rid, rgb, wcs)
        )
        self._composite_worker.composite_failed.connect(self._on_composite_failed)
        self._composite_worker.start()

    def _on_composite_ready(self, report_id, rgb, wcs):
        self._composite_rgb = rgb
        self._composite_wcs = wcs
        self._composite_report_id = report_id
        # only enable if still viewing same report
        if self._current_report_id == report_id:
            self._field_btn.setEnabled(True)

    def _on_composite_failed(self):
        logger.warning("composite build failed, field view unavailable")
        self._field_btn.setEnabled(False)

    def _draw_composite(self):
        self._comp_fig.clear()
        self._comp_ax = self._comp_fig.add_subplot(111)
        self._comp_ax.imshow(self._composite_rgb, origin="lower")

        if self._composite_wcs is not None:
            from astropy.coordinates import SkyCoord
            h, w = self._composite_rgb.shape[:2]
            for c in self._candidates:
                if c.classification == "known" and not self._known_visible:
                    continue
                try:
                    coord = SkyCoord(c.ra, c.dec, unit="deg")
                    px, py = self._composite_wcs.world_to_pixel(coord)
                    px, py = float(px), float(py)
                    if px < 0 or py < 0 or px >= w or py >= h:
                        continue
                    if c.classification == "known":
                        color = _COLORS["known"]
                    else:
                        color = _conf_color(c.confidence)
                    radius = max(3, min(c.snr * 0.8, 15))
                    self._comp_ax.plot(px, py, 'o', markersize=radius,
                                       markerfacecolor='none', markeredgecolor=color,
                                       markeredgewidth=1.0)
                except Exception:
                    continue

        self._comp_ax.set_axis_off()
        self._comp_fig.tight_layout()
        self._comp_canvas.draw()

    def _draw_both(self):
        # render scatter to pixmap
        scatter_pm = self._fig_to_pixmap(self._fig)
        comp_pm = self._fig_to_pixmap(self._comp_fig)

        # scale to half available width
        avail = self.width() // 2 - 10
        if avail > 50:
            self._both_left.setPixmap(
                scatter_pm.scaledToWidth(avail, Qt.TransformationMode.SmoothTransformation)
            )
            self._both_right.setPixmap(
                comp_pm.scaledToWidth(avail, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            self._both_left.setPixmap(scatter_pm)
            self._both_right.setPixmap(comp_pm)

    def _fig_to_pixmap(self, fig):
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        w, h = fig.canvas.get_width_height()
        img = QImage(buf, w, h, QImage.Format.Format_RGBA8888)
        return QPixmap.fromImage(img)

    def _cycle_mode(self):
        if self._view_mode == _MODE_SCATTER:
            self._view_mode = _MODE_COMPOSITE
        elif self._view_mode == _MODE_COMPOSITE:
            self._view_mode = _MODE_BOTH
        else:
            self._view_mode = _MODE_SCATTER
        self._update_mode_ui()

    def _update_mode_ui(self):
        if self._view_mode == _MODE_SCATTER:
            self._stack.setCurrentIndex(1)
            self._field_btn.setText("Field")
            for btn in self._zoom_btns:
                btn.setVisible(True)
        elif self._view_mode == _MODE_COMPOSITE:
            self._draw_composite()
            self._stack.setCurrentIndex(2)
            self._comp_field_btn.setText("Scatter")
            for btn in self._zoom_btns:
                btn.setVisible(False)
        else:
            self._draw_composite()
            self._draw_both()
            self._stack.setCurrentIndex(3)
            self._both_field_btn.setText("Both")
            for btn in self._zoom_btns:
                btn.setVisible(False)

    def set_known_visible(self, visible):
        self._known_visible = visible
        if self._scatter_known is None:
            return
        self._scatter_known.set_visible(visible)
        self._scatter_known.set_label("known" if visible else "known (hidden)")
        self._ax.legend(fontsize=8, loc="upper right")
        self._canvas.draw()

    def _on_scroll(self, event):
        if event.inaxes != self._ax:
            return
        factor = 1.15
        if event.button == "up":
            scale = 1 / factor
        elif event.button == "down":
            scale = factor
        else:
            return

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        self._ax.set_xlim(xdata - new_w * relx, xdata + new_w * (1 - relx))
        self._ax.set_ylim(ydata - new_h * rely, ydata + new_h * (1 - rely))
        self._canvas.draw()

    def _apply_zoom(self, scale):
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        if self._selected_candidate is not None:
            cx = self._selected_candidate.ra
            cy = self._selected_candidate.dec
        else:
            cx = (xlim[0] + xlim[1]) / 2
            cy = (ylim[0] + ylim[1]) / 2
        hw = (xlim[1] - xlim[0]) * scale / 2
        hh = (ylim[1] - ylim[0]) * scale / 2
        self._ax.set_xlim(cx - hw, cx + hw)
        self._ax.set_ylim(cy - hh, cy + hh)
        self._canvas.draw()

    def _zoom_in(self):
        self._apply_zoom(1 / 1.15)

    def _zoom_out(self):
        self._apply_zoom(1.15)

    def _zoom_reset(self):
        if self._original_xlim is not None:
            self._ax.set_xlim(self._original_xlim)
            self._ax.set_ylim(self._original_ylim)
            self._canvas.draw()

    def _on_click(self, event):
        if event.dblclick:
            if event.button != 1 or event.inaxes != self._ax:
                return
            if not self._candidates:
                return
            best = None
            best_dist = float("inf")
            for c in self._candidates:
                if c.classification == "known" and self._scatter_known and not self._scatter_known.get_visible():
                    continue
                px, py = self._ax.transData.transform((c.ra, c.dec))
                dx = px - event.x
                dy = py - event.y
                dist = (dx*dx + dy*dy)**0.5
                if dist < best_dist:
                    best_dist = dist
                    best = c
            if best is not None and best_dist <= 10:
                self.candidate_inspected.emit(best.id)
            return
        if event.button != 1 or event.inaxes != self._ax:
            return

        if not self._candidates:
            return

        best = None
        best_dist = float("inf")
        for c in self._candidates:
            if c.classification == "known" and self._scatter_known and not self._scatter_known.get_visible():
                continue
            px, py = self._ax.transData.transform((c.ra, c.dec))
            dx = px - event.x
            dy = py - event.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = c

        if best is not None and best_dist <= 10:
            self._selected_candidate = best
            self.candidate_selected.emit(best.id)
            if self._selected_marker is not None:
                self._selected_marker.remove()
            self._selected_marker = self._ax.scatter(
                [best.ra], [best.dec], s=120,
                facecolors="none", edgecolors="white", linewidth=2, zorder=10,
            )
            self._canvas.draw()
        else:
            self._selected_candidate = None
            if self._selected_marker is not None:
                self._selected_marker.remove()
                self._selected_marker = None
                self._canvas.draw()
            self.candidate_deselected.emit()

    def deselect(self):
        self._selected_candidate = None
        if self._selected_marker is not None:
            self._selected_marker.remove()
            self._selected_marker = None
            self._canvas.draw()
        self.candidate_deselected.emit()

    def refresh_bookmarks(self):
        from parallax import catalog
        for c in self._candidates:
            fresh = catalog.get(c.id)
            if fresh is not None:
                c.tags = fresh.tags

        if self._scatter_bookmarked is not None:
            self._scatter_bookmarked.remove()
            self._scatter_bookmarked = None

        bm = [c for c in self._candidates if "bookmarked" in (c.tags or [])]
        if bm:
            bm_ras = [c.ra for c in bm]
            bm_decs = [c.dec for c in bm]
            bm_snrs = np.array([c.snr for c in bm])
            bm_sizes = np.clip(bm_snrs * 12, 30, 120)
            self._scatter_bookmarked = self._ax.scatter(
                bm_ras, bm_decs, s=bm_sizes, c="#f1c40f",
                label="bookmarked", alpha=0.9, edgecolors="#c8a000",
                linewidth=1.0, zorder=5,
            )

        self._ax.legend(fontsize=8, loc="upper right")
        self._canvas.draw()

    def select_candidate(self, candidate_id):
        cand = None
        for c in self._candidates:
            if c.id == candidate_id:
                cand = c
                break

        if cand is None:
            self.candidate_selected.emit(candidate_id)
            return

        self._selected_candidate = cand

        if self._selected_marker is not None:
            self._selected_marker.remove()
        self._selected_marker = self._ax.scatter(
            [cand.ra], [cand.dec], s=120,
            facecolors="none", edgecolors="white", linewidth=2, zorder=10,
        )

        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        hw = (xlim[1] - xlim[0]) / 2
        hh = (ylim[1] - ylim[0]) / 2
        self._ax.set_xlim(cand.ra - hw, cand.ra + hw)
        self._ax.set_ylim(cand.dec - hh, cand.dec + hh)

        self._canvas.draw()
        self.candidate_selected.emit(candidate_id)

    def clear(self):
        self._ax.clear()
        self._candidates = []
        self._scatter_known = None
        self._scatter_bookmarked = None
        self._scatter_other = {}
        self._selected_marker = None
        self._canvas.draw()
