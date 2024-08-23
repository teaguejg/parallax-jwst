import os
from datetime import datetime, UTC
from unittest.mock import patch

import pytest

from parallax.types import Candidate, Report


def _make_report(n_cands=3, target="NGC 1234"):
    cands = []
    classes = ["unverified", "known", "known"]
    for i in range(n_cands):
        cands.append(Candidate(
            id=f"cnd_chart{i:04d}", ra=83.82 + i * 0.001, dec=-5.39 + i * 0.001,
            flux=100.0 + i * 10, snr=3.0 + i, classification=classes[i % 3],
            report_id="rpt_20260101_chart", pixel_coords=(50.0, 50.0),
            created_at=datetime.now(UTC),
        ))
    return Report(
        id="rpt_20260101_chart", target=target, instrument="NIRCAM",
        filters=["F200W"],
        created_at=datetime.now(UTC), candidates=cands,
        n_sources_detected=n_cands, n_catalog_matched=2, n_unverified=1,
    )


class TestPlot:
    @patch("matplotlib.pyplot.show")
    def test_runs_ok(self, mock_show, tmp_db):
        from parallax.chart import plot
        rpt = _make_report()
        plot(rpt)

    def test_writes_file(self, tmp_db):
        from parallax.chart import plot
        rpt = _make_report()
        out = os.path.join(tmp_db, "plot.png")
        plot(rpt, output_path=out)
        assert os.path.isfile(out)

    def test_empty_raises(self, tmp_db):
        from parallax.chart import plot
        rpt = _make_report(0)
        with pytest.raises(ValueError):
            plot(rpt)

    def test_show_known(self, tmp_db):
        from parallax.chart import plot
        rpt = _make_report()
        out = os.path.join(tmp_db, "known.png")
        plot(rpt, show_known=True, output_path=out)
        assert os.path.isfile(out)


class TestOverlay:
    def test_single_report(self, tmp_db):
        from parallax.chart import overlay
        out = os.path.join(tmp_db, "overlay.png")
        overlay([_make_report()], output_path=out)
        assert os.path.isfile(out)

    def test_empty_raises(self, tmp_db):
        from parallax.chart import overlay
        with pytest.raises(ValueError):
            overlay([])


class TestField:
    @patch("astroquery.skyview.SkyView.get_images", side_effect=ConnectionError("unavailable"))
    def test_fallback(self, mock_sv, tmp_db):
        from parallax.chart import field
        out = os.path.join(tmp_db, "field.png")
        field(83.82, -5.39, 0.1, output_path=out)
        assert os.path.isfile(out)

    def test_with_candidates(self, tmp_db):
        from parallax.chart import field
        rpt = _make_report()
        out = os.path.join(tmp_db, "fieldcands.png")
        field(83.82, -5.39, 0.1, candidates=rpt.candidates, output_path=out)
        assert os.path.isfile(out)
