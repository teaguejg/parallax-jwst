import os
import tempfile
from datetime import datetime, UTC
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from astropy.io import fits

from conftest import make_fits
from parallax.types import Candidate, Report, CutoutView, ViewSession


def _setup_session(tmpdir):
    hdul = make_fits(n_sources=3, noise=0.05)
    fits_path = os.path.join(tmpdir, "view_test.fits")
    hdul.writeto(fits_path, overwrite=True)

    cand = Candidate(
        id="cnd_view0001", ra=83.8221, dec=-5.3911, flux=100.0, snr=5.0,
        classification="unverified", report_id="rpt_20260101_aabb",
        pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
    )
    rpt = Report(
        id="rpt_20260101_aabb", target="test", instrument="NIRCAM",
        filters=["F200W"],
        created_at=datetime.now(UTC), candidates=[cand],
        n_sources_detected=3, n_catalog_matched=0, n_unverified=1,
    )
    hdul_open = fits.open(fits_path)
    session = ViewSession(rpt, hdul_open)
    return session, cand, fits_path


class TestExamine:
    def test_returns_cutout(self, tmp_db):
        session, cand, _ = _setup_session(tmp_db)
        from parallax.view import examine
        cv = examine(cand, session)
        expected = 60  # config cutout_size
        assert isinstance(cv, CutoutView)
        assert cv.shape == (expected, expected)
        session.fits.close()

    def test_unknown_id_raises(self, tmp_db):
        session, _, _ = _setup_session(tmp_db)
        from parallax.view import examine
        with pytest.raises(ValueError):
            examine("cnd_nonexist", session)
        session.fits.close()

    def test_by_id_string(self, tmp_db):
        session, cand, _ = _setup_session(tmp_db)
        from parallax.view import examine
        cv = examine(cand.id, session)
        assert cv.candidate.id == cand.id
        session.fits.close()


class TestShow:
    def test_writes_file(self, tmp_db):
        session, cand, _ = _setup_session(tmp_db)
        from parallax.view import examine, show
        cv = examine(cand, session)
        out = os.path.join(tmp_db, "cutout.png")
        show(cv, output_path=out)
        assert os.path.isfile(out)
        session.fits.close()

    def test_bad_stretch(self, tmp_db):
        session, cand, _ = _setup_session(tmp_db)
        from parallax.view import examine, show
        cv = examine(cand, session)
        with pytest.raises(ValueError):
            show(cv, stretch="cubic")
        session.fits.close()

    def test_all_nan_cutout_raises(self, tmp_db):
        session, cand, _ = _setup_session(tmp_db)
        from parallax.view import show
        nan_data = np.full((60, 60), np.nan)
        cv = CutoutView(cand, nan_data, None, "fake.fits", (60, 60))
        with pytest.raises(ValueError, match="no valid pixels"):
            show(cv, output_path=os.path.join(tmp_db, "nan.png"))


class TestCompare:
    def test_empty_raises(self, tmp_db):
        from parallax.view import compare
        with pytest.raises(ValueError):
            compare([])

    def test_two_candidates(self, tmp_db):
        session, cand, _ = _setup_session(tmp_db)
        cand2 = Candidate(
            id="cnd_view0002", ra=83.8221, dec=-5.3911, flux=80.0, snr=4.0,
            classification="known", report_id="rpt_20260101_aabb",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
        )
        session.candidates.append(cand2)

        from parallax.view import compare
        out = os.path.join(tmp_db, "compare.png")
        compare([cand, cand2], session, output_path=out)
        assert os.path.isfile(out)
        session.fits.close()
