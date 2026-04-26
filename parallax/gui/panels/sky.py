import logging
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PyQt6.QtCore import pyqtSignal, Qt, QThread
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


class SkyCompositeWorker(QThread):
    """Build a reprojected coadd from report FITS files."""
    sky_ready = pyqtSignal(np.ndarray, object)  # (image_2d, wcs)
    sky_failed = pyqtSignal()

    def __init__(self, report_id, candidates=None, parent=None):
        super().__init__(parent)
        self._report_id = report_id
        self._candidates = candidates or []

    def _field_center(self):
        ras = [c.ra for c in self._candidates if not np.isnan(c.ra)]
        decs = [c.dec for c in self._candidates if not np.isnan(c.dec)]
        if not ras or not decs:
            return None
        return float(np.median(ras)), float(np.median(decs))

    def _pick_path(self, paths, field_coord):
        if len(paths) == 1 or field_coord is None:
            return paths[0]
        import warnings as _w
        from astropy.wcs import FITSFixedWarning
        for path in paths:
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore", FITSFixedWarning)
                    with fits.open(path) as hdul:
                        sci = _find_sci_hdu(hdul)
                        if sci is None:
                            continue
                        w = WCS(sci.header)
                        if w.footprint_contains(field_coord):
                            return path
            except Exception:
                continue
        return paths[0]

    def _load_tiles(self, paths):
        tiles = []
        for path in paths:
            try:
                with fits.open(path) as hdul:
                    sci = _find_sci_hdu(hdul)
                    if sci is None:
                        continue
                    arr = sci.data.astype(np.float64)[::4, ::4]
                    w = WCS(sci.header)
                    w.wcs.crpix = [
                        (w.wcs.crpix[0] - 1) / 4 + 1,
                        (w.wcs.crpix[1] - 1) / 4 + 1,
                    ]
                    if w.wcs.has_cd():
                        w.wcs.cd = w.wcs.cd * 4
                    else:
                        w.wcs.cdelt = [
                            w.wcs.cdelt[0] * 4,
                            w.wcs.cdelt[1] * 4,
                        ]
                    if hasattr(w, 'pixel_shape') and w.pixel_shape is not None:
                        w.pixel_shape = arr.shape[::-1]
                    w.wcs.set()
                    tiles.append((arr, w))
            except Exception:
                continue
        return tiles

    def _normalize(self, coadd):
        interval = ZScaleInterval()
        stretch = AsinhStretch()
        norm = ImageNormalize(coadd, interval=interval, stretch=stretch)
        return np.nan_to_num(norm(coadd), nan=0.0)

    def run(self):
        try:
            from parallax import archive
            fits_map = archive.get_fits_for_report(self._report_id)
            if not fits_map:
                self.sky_failed.emit()
                return

            # pick filter with the most files
            best_filt = max(fits_map, key=lambda f: len(fits_map[f]))
            paths = fits_map[best_filt]
            tiles = self._load_tiles(paths)
            if not tiles:
                self.sky_failed.emit()
                return

            if len(tiles) == 1:
                arr, wcs_out = tiles[0]
                self.sky_ready.emit(self._normalize(np.nan_to_num(arr, nan=0.0)), wcs_out)
                return

            # multi-detector mosaic
            try:
                from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
                from reproject import reproject_interp
                input_data = [(arr, w) for arr, w in tiles]
                wcs_out, shape_out = find_optimal_celestial_wcs(
                    input_data, auto_rotate=False,
                )
                coadd, _ = reproject_and_coadd(
                    input_data, wcs_out, shape_out=shape_out,
                    reproject_function=reproject_interp,
                    combine_function="mean",
                )
                self.sky_ready.emit(self._normalize(np.nan_to_num(coadd, nan=0.0)), wcs_out)
            except Exception:
                logger.warning("mosaic failed, using single tile")
                arr, wcs_out = tiles[0]
                self.sky_ready.emit(self._normalize(np.nan_to_num(arr, nan=0.0)), wcs_out)
        except Exception:
            logger.exception("sky composite worker failed")
            self.sky_failed.emit()


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

        # page 0: progress / idle
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

        # page 1: sky view (WCS composite or scatter fallback)
        self._plot_page = QWidget()
        plot_layout = QVBoxLayout(self._plot_page)
        plot_layout.setContentsMargins(0, 0, 0, 0)

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

        plot_layout.addWidget(self._zoom_row_widget)

        self._fig = Figure(figsize=(8, 8))
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        plot_layout.addWidget(self._canvas)
        self._stack.addWidget(self._plot_page)

        # page 2: loading label while composite builds
        self._loading_page = QWidget()
        ld_layout = QVBoxLayout(self._loading_page)
        ld_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label = QLabel("Loading sky view...")
        self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._loading_label.setStyleSheet("color: #888; font-size: 13px;")
        ld_layout.addWidget(self._loading_label)
        self._stack.addWidget(self._loading_page)

        self._canvas.mpl_connect("button_press_event", self._on_click)
        self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
        self._canvas.mpl_connect("button_release_event", self._on_mouse_release)

        self._candidates = []
        self._scatter_known = None
        self._scatter_other = {}
        self._scatter_bookmarked = None
        self._scatter_viewed = None
        self._selected_marker = None
        self._current_report_id = None
        self._original_xlim = None
        self._original_ylim = None
        self._selected_candidate = None
        self._pan_start = None
        self._pan_xlim = None
        self._pan_ylim = None

        self._zoom_label = None
        self._sky_wcs = None
        self._sky_image = None
        self._wcs_mode = False
        self._sky_worker = None
        self._pending_report = None
        self._layer_vis = {
            "unverified": True, "known": False,
            "bookmarked": True, "viewed": False,
            "conf_high": True, "conf_med": True, "conf_low": True,
        }

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
        self._stack.setCurrentIndex(1)

    def load_report(self, report):
        self._current_report_id = report.id
        self._candidates = list(report.candidates) if report.candidates else []
        self._scatter_known = None
        self._scatter_other = {}
        self._scatter_bookmarked = None
        self._scatter_viewed = None
        self._selected_marker = None
        self._selected_candidate = None
        self._pending_report = report

        # show loading page while worker builds the composite
        self._stack.setCurrentIndex(2)
        self._start_sky_worker(report.id)

    def _start_sky_worker(self, report_id):
        if self._sky_worker is not None and self._sky_worker.isRunning():
            return
        self._sky_worker = SkyCompositeWorker(report_id, self._candidates, self)
        self._sky_worker.sky_ready.connect(
            lambda img, wcs, rid=report_id: self._on_sky_ready(rid, img, wcs)
        )
        self._sky_worker.sky_failed.connect(
            lambda rid=report_id: self._on_sky_failed(rid)
        )
        self._sky_worker.start()

    def _on_sky_ready(self, report_id, image, wcs):
        if self._current_report_id != report_id:
            return
        self._sky_image = image
        self._sky_wcs = wcs
        self._wcs_mode = True
        self._draw_wcs_view()
        self._stack.setCurrentIndex(1)

    def _on_sky_failed(self, report_id):
        if self._current_report_id != report_id:
            return
        logger.warning("sky composite unavailable, falling back to scatter")
        self._sky_wcs = None
        self._sky_image = None
        self._wcs_mode = False
        self._draw_scatter()
        self._stack.setCurrentIndex(1)

    def _draw_wcs_view(self):
        # preserve zoom across redraws (overlay refresh, bookmark toggle)
        saved_xlim = None
        saved_ylim = None
        if self._original_xlim is not None:
            saved_xlim = self._ax.get_xlim()
            saved_ylim = self._ax.get_ylim()

        self._selected_marker = None
        self._zoom_label = None
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)

        from PyQt6.QtWidgets import QApplication
        palette = QApplication.instance().palette()
        bg = palette.color(palette.ColorRole.Window)
        bg_hex = bg.name()
        self._fig.set_facecolor(bg_hex)
        self._ax.set_facecolor("#1a1a1a")
        lightness = bg.lightness()
        txt_color = "white" if lightness < 128 else "black"
        self._ax.tick_params(colors=txt_color)
        self._ax.xaxis.label.set_color(txt_color)
        self._ax.yaxis.label.set_color(txt_color)
        self._ax.title.set_color(txt_color)

        self._ax.imshow(self._sky_image, origin="lower", cmap="gray_r",
                        interpolation="nearest", aspect="auto")
        self._ax.set_xlabel("RA")
        self._ax.set_ylabel("Dec")
        self._set_wcs_ticks()

        if self._pending_report:
            rpt = self._pending_report
            date_str = rpt.created_at.strftime("%Y-%m-%d") if rpt.created_at else ""
            self._ax.set_title(f"{rpt.target}  {date_str}", fontsize=10)

        self._overlay_markers_wcs()

        try:
            h, w = self._sky_image.shape[:2]
            cx, cy = w / 2, h / 2
            ra_c, dec_c = self._sky_wcs.all_pix2world([cx], [cy], 0)
            px_n, py_n = self._sky_wcs.all_world2pix(
                [float(ra_c[0])], [float(dec_c[0]) + 0.01], 0)
            dx = float(px_n[0]) - cx
            dy = float(py_n[0]) - cy
            length = (dx ** 2 + dy ** 2) ** 0.5
            if length > 0:
                dx, dy = dx / length, dy / length
                arrow_len = 0.06
                base_x, base_y = 0.93, 0.08
                tip_x = base_x + dx * arrow_len
                tip_y = base_y + dy * arrow_len
                self._ax.annotate(
                    '', xy=(tip_x, tip_y), xytext=(base_x, base_y),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.5),
                    zorder=20,
                )
                self._ax.text(
                    tip_x + dx * 0.02, tip_y + dy * 0.02, 'N',
                    transform=self._ax.transAxes,
                    ha='center', va='center', fontsize=9,
                    color='white', fontweight='bold', zorder=20,
                )
        except Exception:
            pass

        self._ax.invert_xaxis()

        self._original_xlim = self._ax.get_xlim()
        self._original_ylim = self._ax.get_ylim()
        self._zoom_label = self._ax.text(
            0.99, 0.01, '', transform=self._ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='white', zorder=20,
        )
        self._fig.subplots_adjust(left=0.15, right=0.95, top=0.93, bottom=0.12)
        self._canvas.draw()

        if saved_xlim is not None:
            self._ax.set_xlim(saved_xlim)
            self._ax.set_ylim(saved_ylim)
            self._canvas.draw()

    def _set_wcs_ticks(self):
        if self._sky_wcs is None:
            return
        h, w = self._sky_image.shape[:2]
        try:
            xticks = np.linspace(0, w - 1, 5)
            yticks = np.linspace(0, h - 1, 5)
            xworld = self._sky_wcs.all_pix2world(xticks, np.full(5, h / 2), 0)
            yworld = self._sky_wcs.all_pix2world(np.full(5, w / 2), yticks, 0)
            self._ax.set_xticks(xticks)
            self._ax.set_xticklabels([f"{r:.4f}" for r in xworld[0]], fontsize=7)
            self._ax.set_yticks(yticks)
            self._ax.set_yticklabels([f"{d:.4f}" for d in yworld[1]], fontsize=7)
        except Exception:
            pass

    def _overlay_markers_wcs(self):
        if self._sky_wcs is None:
            return
        h, w = self._sky_image.shape[:2]

        vis_candidates = []
        for c in self._candidates:
            if not self._is_visible(c):
                continue
            vis_candidates.append(c)

        if not vis_candidates:
            return

        ras = np.array([c.ra for c in vis_candidates])
        decs = np.array([c.dec for c in vis_candidates])

        try:
            pxs, pys = self._sky_wcs.all_world2pix(ras, decs, 0)
        except Exception:
            return

        bkt_x = {"known": [], "high": [], "med": [], "low": [], "bm": [], "vw": []}
        bkt_y = {"known": [], "high": [], "med": [], "low": [], "bm": [], "vw": []}
        bkt_s = {"known": [], "high": [], "med": [], "low": [], "bm": [], "vw": []}

        for i, c in enumerate(vis_candidates):
            px, py = float(pxs[i]), float(pys[i])
            if px < -50 or py < -50 or px >= w + 50 or py >= h + 50:
                continue

            if c.classification == "known":
                ms = max(3, min(c.snr * 0.6, 10))
                main = "known"
            else:
                ms = max(3, min(c.snr * 0.8, 15))
                conf = c.confidence
                if conf >= 0.75:
                    main = "high"
                elif conf >= 0.50:
                    main = "med"
                else:
                    main = "low"

            bkt_x[main].append(px)
            bkt_y[main].append(py)
            bkt_s[main].append(ms ** 2)

            tags = c.tags or []
            if "bookmarked" in tags:
                bkt_x["bm"].append(px)
                bkt_y["bm"].append(py)
                bkt_s["bm"].append((ms + 2) ** 2)
            if "viewed" in tags:
                bkt_x["vw"].append(px)
                bkt_y["vw"].append(py)
                bkt_s["vw"].append(max(1, ms - 1) ** 2)

        for key, color, alpha in (
            ("known", _COLORS["known"], 0.6),
            ("high",  "#c0392b",        0.8),
            ("med",   "#e67e22",        0.8),
            ("low",   "#7f8c8d",        0.8),
        ):
            if bkt_x[key]:
                self._ax.scatter(bkt_x[key], bkt_y[key], s=bkt_s[key],
                                 facecolors="none", edgecolors=color,
                                 linewidths=1.0, alpha=alpha, zorder=4)

        if bkt_x["vw"]:
            self._ax.scatter(bkt_x["vw"], bkt_y["vw"], s=bkt_s["vw"],
                             facecolors="none", edgecolors="#27ae60",
                             linewidths=1.0, alpha=0.8, zorder=5)

        if bkt_x["bm"]:
            self._ax.scatter(bkt_x["bm"], bkt_y["bm"], s=bkt_s["bm"],
                             facecolors="none", edgecolors="#f1c40f",
                             linewidths=1.2, alpha=0.9, zorder=6)

        legend_items = []
        if self._layer_vis.get("unverified"):
            if self._layer_vis.get("conf_high", True):
                legend_items.append(("unverified (high)", "#c0392b"))
            if self._layer_vis.get("conf_med", True):
                legend_items.append(("unverified (med)", "#e67e22"))
            if self._layer_vis.get("conf_low", True):
                legend_items.append(("unverified (low)", "#7f8c8d"))
        if self._layer_vis.get("known"):
            legend_items.append(("known", "#4a90d9"))
        if self._layer_vis.get("bookmarked"):
            legend_items.append(("bookmarked", "#f1c40f"))
        if self._layer_vis.get("viewed"):
            legend_items.append(("viewed", "#27ae60"))

        for label, color in legend_items:
            self._ax.scatter([], [], s=40, facecolors="none",
                             edgecolors=color, linewidths=1.0,
                             label=label)
        if legend_items:
            self._ax.legend(fontsize=7, loc="upper right",
                            framealpha=0.5, edgecolor="none")

    def _draw_scatter(self):
        self._fig.clear()
        self._ax = self._fig.add_subplot(111)
        self._wcs_mode = False

        from PyQt6.QtWidgets import QApplication
        palette = QApplication.instance().palette()
        bg = palette.color(palette.ColorRole.Window)
        bg_hex = bg.name()
        self._fig.set_facecolor(bg_hex)
        self._ax.set_facecolor(bg_hex)
        lightness = bg.lightness()
        txt_color = "white" if lightness < 128 else "black"
        self._ax.tick_params(colors=txt_color)
        self._ax.xaxis.label.set_color(txt_color)
        self._ax.yaxis.label.set_color(txt_color)
        self._ax.title.set_color(txt_color)

        for cls, color in _COLORS.items():
            subset = [c for c in self._candidates
                      if c.classification == cls and self._is_visible(c)]
            if not subset:
                continue
            ras = [c.ra for c in subset]
            decs = [c.dec for c in subset]
            snrs = np.array([c.snr for c in subset])
            if cls == "known":
                sizes = np.clip(snrs * 10, 20, 40)
                sc = self._ax.scatter(
                    ras, decs, s=sizes, c=color,
                    label=cls, alpha=0.6, edgecolors="none", linewidth=0,
                )
                self._scatter_known = sc
            else:
                sizes = np.clip(snrs * 10, 20, 200)
                colors = [_conf_color(c.confidence) for c in subset]
                sc = self._ax.scatter(
                    ras, decs, s=sizes, c=colors,
                    label="_nolegend_", alpha=0.8, edgecolors="k", linewidth=0.5,
                )
                self._scatter_other[cls] = sc

        for label, lc in [
            ("unverified (high)", "#c0392b"),
            ("unverified (med)", "#e67e22"),
            ("unverified (low)", "#7f8c8d"),
        ]:
            self._ax.scatter([], [], s=40, c=lc,
                             label=label, edgecolors="k", linewidth=0.5)

        bm = [c for c in self._candidates
              if "bookmarked" in (c.tags or []) and self._is_visible(c)]
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

        vw = [c for c in self._candidates
              if "viewed" in (c.tags or []) and self._is_visible(c)]
        if vw:
            vw_ras = [c.ra for c in vw]
            vw_decs = [c.dec for c in vw]
            vw_snrs = np.array([c.snr for c in vw])
            vw_sizes = np.clip(vw_snrs * 8, 20, 60)
            self._scatter_viewed = self._ax.scatter(
                vw_ras, vw_decs, s=vw_sizes, c="#27ae60",
                label="viewed", alpha=0.8, edgecolors="#1e8449",
                linewidth=0.8, zorder=4,
            )

        self._ax.invert_xaxis()
        self._ax.set_xlabel("RA (deg)")
        self._ax.set_ylabel("Dec (deg)")

        if self._pending_report:
            rpt = self._pending_report
            date_str = rpt.created_at.strftime("%Y-%m-%d") if rpt.created_at else ""
            self._ax.set_title(f"{rpt.target}  {date_str}")
            if rpt.filters:
                fstr = " ".join(sorted(set(rpt.filters)))
                self._ax.annotate(
                    fstr, xy=(0.99, 0.01), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=7, color="#555555",
                )

        self._ax.legend(fontsize=8, loc="upper right")
        self._ax.grid(True, alpha=0.3)
        self._original_xlim = self._ax.get_xlim()
        self._original_ylim = self._ax.get_ylim()
        self._fig.tight_layout()
        self._canvas.draw()

    def _is_visible(self, c):
        tags = c.tags or []
        is_bm = "bookmarked" in tags
        is_vw = "viewed" in tags

        if is_bm and self._layer_vis.get("bookmarked"):
            return True
        if is_vw and self._layer_vis.get("viewed"):
            return True
        if c.classification == "known":
            return self._layer_vis.get("known", False)
        # unverified with no bookmarked/viewed tags
        if not is_bm and not is_vw:
            if not self._layer_vis.get("unverified", False):
                return False
            conf = c.confidence
            if conf >= 0.75:
                return self._layer_vis.get("conf_high", True)
            elif conf >= 0.50:
                return self._layer_vis.get("conf_med", True)
            else:
                return self._layer_vis.get("conf_low", True)
        return False

    def set_layer_visibility(self, layers):
        self._layer_vis = dict(layers)
        if self._wcs_mode:
            self._draw_wcs_view()
        elif self._scatter_known is not None:
            # scatter mode: redraw fully
            self._draw_scatter()

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
        if self._original_xlim is not None:
            orig_w = abs(self._original_xlim[1] - self._original_xlim[0])
            orig_h = abs(self._original_ylim[1] - self._original_ylim[0])
            if new_w > orig_w or new_h > orig_h:
                return
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        self._ax.set_xlim(xdata - new_w * relx, xdata + new_w * (1 - relx))
        self._ax.set_ylim(ydata - new_h * rely, ydata + new_h * (1 - rely))
        self._canvas.draw_idle()
        self._update_zoom_label()

    def _apply_zoom(self, scale):
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        if self._selected_candidate is not None and not self._wcs_mode:
            cx = self._selected_candidate.ra
            cy = self._selected_candidate.dec
        elif self._selected_candidate is not None and self._wcs_mode and self._sky_wcs is not None:
            try:
                px, py = self._sky_wcs.all_world2pix(
                    [self._selected_candidate.ra], [self._selected_candidate.dec], 0)
                cx, cy = float(px[0]), float(py[0])
            except Exception:
                cx = (xlim[0] + xlim[1]) / 2
                cy = (ylim[0] + ylim[1]) / 2
        else:
            cx = (xlim[0] + xlim[1]) / 2
            cy = (ylim[0] + ylim[1]) / 2
        hw = (xlim[1] - xlim[0]) * scale / 2
        hh = (ylim[1] - ylim[0]) * scale / 2
        if self._original_xlim is not None:
            orig_hw = abs(self._original_xlim[1] - self._original_xlim[0]) / 2
            orig_hh = abs(self._original_ylim[1] - self._original_ylim[0]) / 2
            if hw > orig_hw or hh > orig_hh:
                return
        self._ax.set_xlim(cx - hw, cx + hw)
        self._ax.set_ylim(cy - hh, cy + hh)
        self._canvas.draw_idle()
        self._update_zoom_label()

    def _zoom_in(self):
        self._apply_zoom(1 / 1.15)

    def _zoom_out(self):
        self._apply_zoom(1.15)

    def _zoom_reset(self):
        if self._original_xlim is not None:
            self._ax.set_xlim(self._original_xlim)
            self._ax.set_ylim(self._original_ylim)
            self._canvas.draw_idle()
        self._update_zoom_label()

    def _update_zoom_label(self):
        if self._zoom_label is None or self._original_xlim is None:
            return
        cur = abs(self._ax.get_xlim()[1] - self._ax.get_xlim()[0])
        orig = abs(self._original_xlim[1] - self._original_xlim[0])
        ratio = orig / cur
        if abs(ratio - 1.0) < 0.05:
            self._zoom_label.set_text('')
        else:
            self._zoom_label.set_text(f'{ratio:.1f}x')
        self._canvas.draw_idle()

    def _candidate_at_event(self, event):
        if not self._candidates:
            return None, float("inf")

        best = None
        best_dist = float("inf")

        for c in self._candidates:
            if not self._is_visible(c):
                continue

            if self._wcs_mode and self._sky_wcs is not None:
                try:
                    px, py = self._sky_wcs.all_world2pix([c.ra], [c.dec], 0)
                    data_x, data_y = float(px[0]), float(py[0])
                except Exception:
                    continue
            else:
                data_x, data_y = c.ra, c.dec

            sx, sy = self._ax.transData.transform((data_x, data_y))
            dx = sx - event.x
            dy = sy - event.y
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = c

        return best, best_dist

    def _on_click(self, event):
        if event.button == 3 and event.inaxes == self._ax and event.xdata is not None:
            self._pan_start = (event.x, event.y)
            self._pan_xlim = self._ax.get_xlim()
            self._pan_ylim = self._ax.get_ylim()
            return

        if event.dblclick:
            if event.button != 1 or event.inaxes != self._ax:
                return
            best, best_dist = self._candidate_at_event(event)
            if best is not None and best_dist <= 10:
                self.candidate_inspected.emit(best.id)
            return

        if event.button != 1 or event.inaxes != self._ax:
            return

        best, best_dist = self._candidate_at_event(event)

        if best is not None and best_dist <= 10:
            self._selected_candidate = best
            self.candidate_selected.emit(best.id)
            self._draw_selection_marker(best)
        else:
            self._selected_candidate = None
            if self._selected_marker is not None:
                self._selected_marker.remove()
                self._selected_marker = None
                self._canvas.draw_idle()
            self.candidate_deselected.emit()

    def _on_mouse_move(self, event):
        if self._pan_start is None or event.button != 3:
            return
        dx_px = event.x - self._pan_start[0]
        dy_px = event.y - self._pan_start[1]
        inv = self._ax.transData.inverted()
        origin = inv.transform((0, 0))
        delta = inv.transform((dx_px, dy_px))
        dx_data = delta[0] - origin[0]
        dy_data = delta[1] - origin[1]
        self._ax.set_xlim(self._pan_xlim[0] - dx_data, self._pan_xlim[1] - dx_data)
        self._ax.set_ylim(self._pan_ylim[0] - dy_data, self._pan_ylim[1] - dy_data)
        self._canvas.draw_idle()

    def _on_mouse_release(self, event):
        if event.button == 3:
            self._pan_start = None
            self._update_zoom_label()

    def _draw_selection_marker(self, cand):
        if self._selected_marker is not None:
            self._selected_marker.remove()

        if self._wcs_mode and self._sky_wcs is not None:
            try:
                px, py = self._sky_wcs.all_world2pix([cand.ra], [cand.dec], 0)
                mx, my = float(px[0]), float(py[0])
            except Exception:
                mx, my = cand.ra, cand.dec
        else:
            mx, my = cand.ra, cand.dec

        self._selected_marker = self._ax.plot(
            mx, my, marker='o', markersize=14,
            markerfacecolor='none', markeredgecolor='white',
            markeredgewidth=2, zorder=10, linestyle='none',
        )[0]
        self._canvas.draw_idle()

    def deselect(self):
        self._selected_candidate = None
        if self._selected_marker is not None:
            self._selected_marker.remove()
            self._selected_marker = None
            self._canvas.draw_idle()
        self.candidate_deselected.emit()

    def refresh_overlays(self):
        from parallax import catalog as cat_mod
        for c in self._candidates:
            fresh = cat_mod.get(c.id)
            if fresh is not None:
                c.tags = fresh.tags

        if self._wcs_mode:
            self._draw_wcs_view()
            return

        # scatter mode refresh
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

        if self._scatter_viewed is not None:
            self._scatter_viewed.remove()
            self._scatter_viewed = None

        vw = [c for c in self._candidates if "viewed" in (c.tags or [])]
        if vw:
            vw_ras = [c.ra for c in vw]
            vw_decs = [c.dec for c in vw]
            vw_snrs = np.array([c.snr for c in vw])
            vw_sizes = np.clip(vw_snrs * 8, 20, 60)
            self._scatter_viewed = self._ax.scatter(
                vw_ras, vw_decs, s=vw_sizes, c="#27ae60",
                label="viewed", alpha=0.8, edgecolors="#1e8449",
                linewidth=0.8, zorder=4,
            )

        self._ax.legend(fontsize=8, loc="upper right")
        self._canvas.draw_idle()

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
        self._draw_selection_marker(cand)

        if self._wcs_mode and self._sky_wcs is not None:
            try:
                px, py = self._sky_wcs.all_world2pix([cand.ra], [cand.dec], 0)
                cx, cy = float(px[0]), float(py[0])
            except Exception:
                cx, cy = None, None
        else:
            cx, cy = cand.ra, cand.dec

        if cx is not None:
            xlim = self._ax.get_xlim()
            ylim = self._ax.get_ylim()
            hw = (xlim[1] - xlim[0]) / 2
            hh = (ylim[1] - ylim[0]) / 2
            self._ax.set_xlim(cx - hw, cx + hw)
            self._ax.set_ylim(cy - hh, cy + hh)
            self._canvas.draw_idle()

        self.candidate_selected.emit(candidate_id)

    def clear(self):
        self._ax.clear()
        self._candidates = []
        self._scatter_known = None
        self._scatter_bookmarked = None
        self._scatter_viewed = None
        self._scatter_other = {}
        self._selected_marker = None
        self._sky_wcs = None
        self._sky_image = None
        self._wcs_mode = False
        self._canvas.draw_idle()
