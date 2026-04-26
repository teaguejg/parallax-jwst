import os
import sys
import types
import tempfile
from datetime import datetime, UTC
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from astropy.wcs import WCS

PyQt6 = pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    try:
        _app = QApplication([])
    except Exception:
        pytest.skip("no display available", allow_module_level=True)


from parallax.types import Candidate, Report
from parallax.gui.panels.sky import SkyPanel, SkyCompositeWorker, _conf_color


def _make_wcs(crval=(83.82, -5.39), shape=(200, 200)):
    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] // 2, shape[0] // 2]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    w.wcs.crval = list(crval)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def _make_candidate(cid, ra, dec, cls="unverified", snr=10.0, conf=0.8,
                    tags=None, report_id="rpt_001"):
    return Candidate(
        id=cid, ra=ra, dec=dec, flux=100.0, snr=snr,
        classification=cls, report_id=report_id,
        pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
        confidence=conf, tags=tags or [],
    )


def _make_report(candidates=None, report_id="rpt_001"):
    return Report(
        id=report_id, target="M92", instrument="NIRCAM",
        filters=["F200W"], created_at=datetime.now(UTC),
        candidates=candidates or [],
    )


class TestFallbackOnNoFits:
    def test_scatter_fallback_empty_fits(self):
        panel = SkyPanel()
        c = _make_candidate("cnd_aaa", 83.82, -5.39)
        report = _make_report([c])

        with patch.object(SkyCompositeWorker, 'start') as mock_start:
            panel.load_report(report)
            panel._on_sky_failed(report.id)

        assert panel._wcs_mode is False
        assert panel._sky_wcs is None
        panel.close()

    def test_scatter_fallback_wrong_report(self):
        """Failure for a stale report_id should be ignored."""
        panel = SkyPanel()
        c = _make_candidate("cnd_bbb", 83.82, -5.39)
        report = _make_report([c])

        with patch.object(SkyCompositeWorker, 'start'):
            panel.load_report(report)
            panel._on_sky_failed("rpt_old")

        assert panel._stack.currentIndex() == 2
        panel.close()


class TestMarkerPositions:
    def test_wcs_pixel_coords(self):
        """Candidate at WCS center should map near image center."""
        wcs = _make_wcs(crval=(83.82, -5.39), shape=(200, 200))
        px, py = wcs.all_world2pix([83.82], [-5.39], 0)
        assert abs(float(px[0]) - 100) < 2
        assert abs(float(py[0]) - 100) < 2

    def test_markers_plotted_on_wcs_view(self):
        panel = SkyPanel()
        c1 = _make_candidate("cnd_001", 83.82, -5.39, cls="unverified", conf=0.8)
        c2 = _make_candidate("cnd_002", 83.82, -5.39, cls="known", snr=5.0)
        report = _make_report([c1, c2])

        wcs = _make_wcs()
        image = np.random.default_rng(0).random((200, 200))

        with patch.object(SkyCompositeWorker, 'start'):
            panel.load_report(report)
            panel._on_sky_ready(report.id, image, wcs)

        assert panel._wcs_mode is True
        assert len(panel._ax.collections) > 0
        panel.close()


class TestClassificationColoring:
    def test_conf_color_high(self):
        assert _conf_color(0.80) == "#c0392b"

    def test_conf_color_med(self):
        assert _conf_color(0.55) == "#e67e22"

    def test_conf_color_low(self):
        assert _conf_color(0.30) == "#7f8c8d"


class TestClickSignal:
    def test_candidate_selected_signal(self):
        panel = SkyPanel()
        c = _make_candidate("cnd_click", 83.82, -5.39)
        report = _make_report([c])

        with patch.object(SkyCompositeWorker, 'start'):
            panel.load_report(report)
            panel._on_sky_failed(report.id)  # scatter fallback

        received = []
        panel.candidate_selected.connect(received.append)

        # simulate select_candidate
        panel.select_candidate("cnd_click")
        assert "cnd_click" in received
        panel.close()

    def test_deselect_signal(self):
        panel = SkyPanel()
        received = []
        panel.candidate_deselected.connect(lambda: received.append(True))
        panel.deselect()
        assert received
        panel.close()


class TestWcsViewSetup:
    def test_ready_sets_wcs_mode(self):
        panel = SkyPanel()
        report = _make_report([_make_candidate("cnd_x", 83.82, -5.39)])

        wcs = _make_wcs()
        image = np.zeros((200, 200))

        with patch.object(SkyCompositeWorker, 'start'):
            panel.load_report(report)
            panel._on_sky_ready(report.id, image, wcs)

        assert panel._wcs_mode is True
        assert panel._sky_wcs is wcs
        assert panel._stack.currentIndex() == 1
        panel.close()

    def test_known_hidden_by_default(self):
        panel = SkyPanel()
        c_known = _make_candidate("cnd_k", 83.82, -5.39, cls="known")
        report = _make_report([c_known])

        wcs = _make_wcs()
        image = np.zeros((200, 200))

        with patch.object(SkyCompositeWorker, 'start'):
            panel.load_report(report)
            panel._on_sky_ready(report.id, image, wcs)

        assert panel._layer_vis.get("known") is False
        panel.close()

    def test_set_layer_visibility_redraws(self):
        panel = SkyPanel()
        c = _make_candidate("cnd_kv", 83.82, -5.39, cls="known")
        report = _make_report([c])

        wcs = _make_wcs()
        image = np.zeros((200, 200))

        with patch.object(SkyCompositeWorker, 'start'):
            panel.load_report(report)
            panel._on_sky_ready(report.id, image, wcs)

        panel.set_layer_visibility({"unverified": True, "known": True,
                                    "bookmarked": True, "viewed": False})
        assert panel._layer_vis.get("known") is True
        assert len(panel._ax.collections) > 0
        panel.close()


def test_clear_resets_state():
    panel = SkyPanel()
    panel._wcs_mode = True
    panel._sky_wcs = _make_wcs()
    panel._sky_image = np.zeros((10, 10))
    panel.clear()
    assert panel._wcs_mode is False
    assert panel._sky_wcs is None
    assert panel._sky_image is None
    panel.close()


class TestLayerVisibility:
    """Unit tests for _is_visible() layer logic."""

    def _panel_with_layers(self, **overrides):
        panel = SkyPanel()
        panel._layer_vis = {
            "unverified": True, "known": False,
            "bookmarked": True, "viewed": False,
            "conf_high": True, "conf_med": True, "conf_low": True,
        }
        panel._layer_vis.update(overrides)
        return panel

    def test_unverified_visible_by_default(self):
        panel = self._panel_with_layers()
        c = _make_candidate("cnd_u1", 83.82, -5.39, cls="unverified")
        assert panel._is_visible(c) is True
        panel.close()

    def test_known_hidden_by_default(self):
        panel = self._panel_with_layers()
        c = _make_candidate("cnd_k1", 83.82, -5.39, cls="known")
        assert panel._is_visible(c) is False
        panel.close()

    def test_known_visible_when_layer_on(self):
        panel = self._panel_with_layers(known=True)
        c = _make_candidate("cnd_k2", 83.82, -5.39, cls="known")
        assert panel._is_visible(c) is True
        panel.close()

    def test_unverified_hidden_when_layer_off(self):
        panel = self._panel_with_layers(unverified=False)
        c = _make_candidate("cnd_u2", 83.82, -5.39, cls="unverified")
        assert panel._is_visible(c) is False
        panel.close()

    def test_bookmarked_unverified_visible_via_bookmark_layer(self):
        # unverified off, bookmarked on - candidate visible via bookmarked
        panel = self._panel_with_layers(unverified=False, bookmarked=True)
        c = _make_candidate("cnd_bm1", 83.82, -5.39, cls="unverified",
                            tags=["bookmarked"])
        assert panel._is_visible(c) is True
        panel.close()

    def test_bookmarked_hidden_when_both_layers_off(self):
        panel = self._panel_with_layers(unverified=False, bookmarked=False)
        c = _make_candidate("cnd_bm2", 83.82, -5.39, cls="unverified",
                            tags=["bookmarked"])
        assert panel._is_visible(c) is False
        panel.close()

    def test_viewed_known_visible_via_viewed_layer(self):
        # known off, viewed on - known candidate with viewed tag is visible
        panel = self._panel_with_layers(known=False, viewed=True)
        c = _make_candidate("cnd_vw1", 83.82, -5.39, cls="known",
                            tags=["viewed"])
        assert panel._is_visible(c) is True
        panel.close()

    def test_no_tags_known_hidden_both_layers_off(self):
        panel = self._panel_with_layers(known=False, bookmarked=False,
                                        viewed=False)
        c = _make_candidate("cnd_k3", 83.82, -5.39, cls="known")
        assert panel._is_visible(c) is False
        panel.close()

    def test_unverified_bookmarked_not_visible_when_only_unverified_on(self):
        # bookmarked candidates belong to the Bookmarked layer, not Unverified
        panel = self._panel_with_layers(unverified=True, bookmarked=False)
        c = _make_candidate("cnd_bm3", 83.82, -5.39, cls="unverified",
                            tags=["bookmarked"])
        assert panel._is_visible(c) is False
        panel.close()

    def test_unverified_viewed_not_visible_when_only_unverified_on(self):
        panel = self._panel_with_layers(unverified=True, viewed=False)
        c = _make_candidate("cnd_vw2", 83.82, -5.39, cls="unverified",
                            tags=["viewed"])
        assert panel._is_visible(c) is False
        panel.close()

    def test_conf_high_off_hides_high_confidence(self):
        panel = self._panel_with_layers(conf_high=False)
        c = _make_candidate("cnd_ch1", 83.82, -5.39, cls="unverified", conf=0.90)
        assert panel._is_visible(c) is False
        panel.close()

    def test_conf_low_off_hides_low_leaves_high_visible(self):
        panel = self._panel_with_layers(conf_low=False)
        lo = _make_candidate("cnd_lo1", 83.82, -5.39, cls="unverified", conf=0.30)
        hi = _make_candidate("cnd_hi1", 83.82, -5.39, cls="unverified", conf=0.85)
        assert panel._is_visible(lo) is False
        assert panel._is_visible(hi) is True
        panel.close()

    def test_unverified_off_ignores_tier_state(self):
        # all tiers on, but Unverified itself is off - nothing should show
        panel = self._panel_with_layers(unverified=False,
                                        conf_high=True, conf_med=True, conf_low=True)
        for conf in (0.90, 0.60, 0.20):
            c = _make_candidate(f"cnd_uoff_{int(conf*100)}",
                                83.82, -5.39, cls="unverified", conf=conf)
            assert panel._is_visible(c) is False
        panel.close()

    def test_set_layer_visibility_updates_dict(self):
        panel = SkyPanel()
        new_state = {"unverified": False, "known": True,
                     "bookmarked": False, "viewed": True}
        panel.set_layer_visibility(new_state)
        assert panel._layer_vis == new_state
        panel.close()
