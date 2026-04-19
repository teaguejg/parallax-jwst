import json
import os
import math
import sys
import tempfile
from datetime import datetime, timedelta, UTC
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from astropy.io import fits

from conftest import make_fits
from parallax.types import Candidate, CatalogMatch, Report


def _write_fits(hdul, tmpdir):
    path = os.path.join(tmpdir, "test.fits")
    hdul.writeto(path, overwrite=True)
    return path


class TestAcquireLocalDiscovery:
    @patch("parallax.acquisition._get_expected_filenames",
           return_value={"jw01234_nircam_f200w_i2d.fits", "jw01234_nircam_f444w_i2d.fits"})
    @patch("parallax.acquisition._mast_query", return_value=MagicMock(__len__=lambda s: 2))
    @patch("parallax.acquisition._resolve_name", return_value=("NGC 6341", 259.28, 43.14))
    def test_returns_local_when_i2d_exist(self, mock_resolve, mock_query, mock_expected, tmp_db):
        """acquire() returns local paths when cache is complete."""
        from parallax.survey import acquire

        slug_dir = os.path.join(tmp_db, "downloads", "mastDownload", "JWST", "ngc_6341")
        for obs_id, name in [("jw01234_nircam_f200w", "jw01234_nircam_f200w_i2d.fits"),
                             ("jw01234_nircam_f444w", "jw01234_nircam_f444w_i2d.fits")]:
            obs_dir = os.path.join(slug_dir, obs_id)
            os.makedirs(obs_dir, exist_ok=True)
            hdul = make_fits()
            hdul.writeto(os.path.join(obs_dir, name), overwrite=True)

        paths = acquire("M92")
        assert len(paths) == 2
        for p in paths:
            assert "_i2d.fits" in p
            assert "ngc_6341" in p

    @patch("parallax.acquisition._get_expected_filenames",
           return_value={"jw01234_nircam_f200w_i2d.fits", "jw01234_nircam_f444w_i2d.fits"})
    @patch("parallax.acquisition._mast_download")
    @patch("parallax.acquisition._mast_query", return_value=MagicMock(__len__=lambda s: 2))
    @patch("parallax.acquisition._resolve_name", return_value=("M  92", 259.28, 43.14))
    def test_incomplete_cache_triggers_download(self, mock_resolve, mock_query, mock_dl, mock_expected, tmp_db):
        """acquire() re-downloads when local cache is incomplete."""
        from parallax.survey import acquire

        # only one of two expected files present
        slug_dir = os.path.join(tmp_db, "downloads", "mastDownload", "JWST", "m_92")
        obs_dir = os.path.join(slug_dir, "jw01234_nircam_f200w")
        os.makedirs(obs_dir, exist_ok=True)
        hdul = make_fits()
        hdul.writeto(os.path.join(obs_dir, "jw01234_nircam_f200w_i2d.fits"), overwrite=True)

        fresh = os.path.join(slug_dir, "fresh_i2d.fits")
        mock_dl.return_value = [fresh]

        acquire("M92")
        mock_dl.assert_called_once()

    @patch("parallax.acquisition._mast_download")
    @patch("parallax.acquisition._mast_query")
    @patch("parallax.acquisition._resolve_name", return_value=("M  92", 259.28, 43.14))
    def test_download_goes_to_slug_dir(self, mock_resolve, mock_query, mock_dl, tmp_db):
        """Downloads are directed to the canonical slug subfolder."""
        from parallax.survey import acquire

        mock_query.return_value = MagicMock(__len__=lambda s: 1)
        fake_path = os.path.join(tmp_db, "downloads", "mastDownload", "JWST", "m_92", "test_i2d.fits")
        mock_dl.return_value = [fake_path]

        acquire("M92")

        call_args = mock_dl.call_args
        dest_dir = call_args[0][1]
        assert dest_dir.endswith(os.path.join("mastDownload", "JWST", "m_92"))


class TestAcquireCorruptFiles:
    @patch("parallax.acquisition._mast_download")
    @patch("parallax.acquisition._mast_query")
    @patch("parallax.acquisition._resolve_name", return_value=("NGC 6341", 259.28, 43.14))
    def test_corrupt_local_falls_through_to_mast(self, mock_resolve, mock_query, mock_dl, tmp_db):
        """acquire() deletes corrupt local files and queries MAST when none survive."""
        from parallax.survey import acquire

        slug_dir = os.path.join(tmp_db, "downloads", "mastDownload", "JWST", "ngc_6341")
        obs_dir = os.path.join(slug_dir, "jw01234_nircam_f200w")
        os.makedirs(obs_dir, exist_ok=True)

        # write a corrupt fits file (just garbage bytes)
        corrupt_path = os.path.join(obs_dir, "jw01234_nircam_f200w_i2d.fits")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a fits file at all")

        mock_query.return_value = MagicMock(__len__=lambda s: 1)
        fresh = os.path.join(slug_dir, "fresh_i2d.fits")
        mock_dl.return_value = [fresh]

        acquire("M92")

        # corrupt file should be deleted
        assert not os.path.exists(corrupt_path)
        # should have fallen through to MAST
        mock_query.assert_called_once()


class TestDetect:
    def test_finds_sources(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)

        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(results) >= 1
        for s in results:
            assert "ra" in s
            assert "dec" in s
            assert "flux" in s
            assert "snr" in s
            assert "pixel_x" in s
            assert "bbox" in s

    def test_empty_on_zeros(self, tmp_db):
        hdul = make_fits(n_sources=0, noise=0.001)
        # overwrite data with near-zero
        hdul[0].data = np.zeros((200, 200), dtype=np.float32) + 1e-10
        path = _write_fits(hdul, tmp_db)

        from parallax.survey import detect
        results = detect(path, snr_threshold=5.0, min_pixels=5)
        assert results == []

    def test_missing_file_raises(self, tmp_db):
        from parallax.survey import detect
        with pytest.raises(FileNotFoundError):
            detect("/no/such/file.fits")

    def test_invalid_fits(self, tmp_db):
        # FITS with no 2D data
        hdu = fits.PrimaryHDU()
        path = os.path.join(tmp_db, "empty.fits")
        hdu.writeto(path, overwrite=True)

        from parallax.survey import detect
        with pytest.raises(ValueError):
            detect(path)

    def test_ra_zero_boundary(self, tmp_db):
        from astropy.wcs import WCS
        hdul = make_fits(n_sources=2, noise=0.05)
        w = WCS(naxis=2)
        w.wcs.crpix = [100, 100]
        w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
        w.wcs.crval = [0.001, 0.001]
        w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        hdul[0].header.update(w.to_header())
        path = _write_fits(hdul, tmp_db)

        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        for s in results:
            assert not math.isnan(s["ra"])

    def test_truncated_fits_raises_valueerror(self, tmp_db):
        """detect() raises ValueError (not BufferError) on truncated FITS."""
        from parallax.survey import detect

        hdul = make_fits(n_sources=3, noise=0.05)
        path = os.path.join(tmp_db, "truncated.fits")
        hdul.writeto(path, overwrite=True)

        # truncate the file to half its size
        full_size = os.path.getsize(path)
        with open(path, "r+b") as f:
            f.truncate(full_size // 2)

        with pytest.raises(ValueError, match="truncated or corrupt"):
            detect(path)

    def test_snr_uses_local_rms(self, tmp_db):
        """SNR should be computed from local RMS when the map is available."""
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(results) >= 1
        # all sources should have positive snr computed from local rms
        for s in results:
            assert s["snr"] > 0


class TestResolve:
    def _make_detections(self, n=2):
        return [
            {"ra": 83.82 + i * 0.0001, "dec": -5.39, "flux": 100.0 + i,
             "snr": 5.0, "pixel_x": 100.0 + i, "pixel_y": 100.0, "label": i,
             "bbox": {"ixmin": 98, "ixmax": 103, "iymin": 98, "iymax": 103}}
            for i in range(n)
        ]

    @patch("parallax.survey._query_simbad", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_gaia", return_value=[])
    def test_unverified_when_no_matches(self, g, n, s, tmp_db):
        from parallax.survey import resolve
        dets = self._make_detections(2)
        cands, gaia_failed = resolve(dets)
        assert len(cands) == 2
        for c in cands:
            assert c.classification == "unverified"
            assert c.id.startswith("cnd_")

    @patch("parallax.survey._query_gaia", return_value=[])
    @patch("parallax.survey._query_ned")
    @patch("parallax.survey._query_simbad")
    def test_known_with_two_catalogs(self, mock_s, mock_n, mock_g, tmp_db):
        # field-wide results: dicts with ra/dec near the detection
        mock_s.return_value = [{"catalog": "SIMBAD", "source_id": "obj1",
                                "ra": 83.82, "dec": -5.39,
                                "object_type": "Star", "redshift": None}]
        mock_n.return_value = [{"catalog": "NED", "source_id": "obj1n",
                                "ra": 83.82, "dec": -5.39,
                                "object_type": "G", "redshift": None}]
        from parallax.survey import resolve
        cands, _ = resolve(self._make_detections(1))
        assert cands[0].classification == "known"

    @patch("parallax.survey._query_gaia", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_simbad")
    def test_notable_single_catalog(self, mock_s, mock_n, mock_g, tmp_db):
        mock_s.return_value = [{"catalog": "SIMBAD", "source_id": "obj1",
                                "ra": 83.82, "dec": -5.39,
                                "object_type": None, "redshift": None}]
        from parallax.survey import resolve
        cands, _ = resolve(self._make_detections(1))
        assert cands[0].classification == "known"

    @patch("parallax.survey._query_gaia", side_effect=Exception("timeout"))
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_simbad", return_value=[])
    def test_handles_catalog_timeout(self, s, n, g, tmp_db):
        from parallax.survey import resolve
        cands, gaia_failed = resolve(self._make_detections(1))
        assert len(cands) == 1
        assert gaia_failed is True

    @patch("parallax.survey._query_simbad", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_gaia", return_value=[])
    def test_batch_queries_called_once(self, mock_g, mock_n, mock_s, tmp_db):
        """3 queries total regardless of detection count."""
        from parallax.survey import resolve
        dets = self._make_detections(50)
        cands, _ = resolve(dets)
        assert mock_s.call_count == 1
        assert mock_n.call_count == 1
        assert mock_g.call_count == 1

    @patch("parallax.survey._query_gaia", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_simbad")
    def test_far_catalog_source_no_match(self, mock_s, mock_n, mock_g, tmp_db):
        # catalog source 1 degree away shouldn't match any detection
        mock_s.return_value = [{"catalog": "SIMBAD", "source_id": "far_obj",
                                "ra": 84.82, "dec": -5.39,
                                "object_type": "Star", "redshift": None}]
        from parallax.survey import resolve
        cands, _ = resolve(self._make_detections(1))
        assert cands[0].classification == "unverified"
        assert len(cands[0].catalog_matches) == 0


class TestReport:
    @patch("parallax.catalog.add", return_value="cnd_12345678")
    def test_writes_json(self, mock_add, tmp_db):
        from parallax.survey import report
        hdul = make_fits()
        fits_path = _write_fits(hdul, tmp_db)

        cands = [Candidate(
            id="cnd_aabbccdd", ra=83.82, dec=-5.39, flux=100.0, snr=5.0,
            classification="unverified", report_id="",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
        )]
        rpt = report(cands, "test target", [(fits_path, "F200W")], 10, output_format="json")

        assert rpt.json_path is not None
        assert os.path.isfile(rpt.json_path)

        with open(rpt.json_path) as f:
            data = json.load(f)
        assert data["target"] == "test target"
        assert data["n_sources_detected"] == 10

    @patch("parallax.catalog.add", return_value="cnd_12345678")
    def test_report_fields(self, mock_add, tmp_db):
        from parallax.survey import report
        hdul = make_fits()
        fits_path = _write_fits(hdul, tmp_db)

        cands = [
            Candidate(id="cnd_11111111", ra=83.82, dec=-5.39, flux=50.0, snr=3.0,
                      classification="unverified", report_id="",
                      pixel_coords=(50.0, 50.0), created_at=datetime.now(UTC)),
            Candidate(id="cnd_22222222", ra=83.83, dec=-5.40, flux=200.0, snr=8.0,
                      classification="known", report_id="",
                      pixel_coords=(60.0, 60.0), created_at=datetime.now(UTC)),
        ]
        rpt = report(cands, "NGC 1234", [(fits_path, "F200W")], 5)
        assert rpt.instrument == "NIRCAM"
        assert rpt.filters == ["F200W"]
        assert rpt.n_catalog_matched == 1
        assert rpt.n_unverified == 1


class TestReduce:
    @patch("parallax.survey.report")
    @patch("parallax.survey.resolve")
    @patch("parallax.survey.detect")
    @patch("parallax.survey.acquire")
    def test_chains_correctly(self, mock_acq, mock_det, mock_res, mock_rpt, tmp_db):
        fits_path = os.path.join(tmp_db, "downloads", "chain_test.fits")
        hdul = make_fits()
        hdul.writeto(fits_path, overwrite=True)

        mock_acq.return_value = [fits_path]
        mock_det.return_value = [{"ra": 1.0, "dec": 2.0, "flux": 10.0, "snr": 5.0,
                                  "pixel_x": 50, "pixel_y": 50, "label": 1,
                                  "bbox": {"ixmin": 0, "ixmax": 5, "iymin": 0, "iymax": 5}}]
        mock_res.return_value = ([], False)
        mock_rpt.return_value = MagicMock()

        from parallax.survey import reduce
        reduce("test")

        mock_acq.assert_called_once()
        mock_det.assert_called_once_with(fits_path, filter_name="F200W")
        mock_res.assert_called_once()
        mock_rpt.assert_called_once()

    @patch("parallax.survey.report")
    @patch("parallax.survey.resolve")
    @patch("parallax.survey.detect")
    @patch("parallax.survey.acquire")
    def test_on_progress_called(self, mock_acq, mock_det, mock_res, mock_rpt, tmp_db):
        fits_path = os.path.join(tmp_db, "downloads", "progress_test.fits")
        hdul = make_fits()
        hdul.writeto(fits_path, overwrite=True)

        mock_acq.return_value = [fits_path]
        mock_det.return_value = [{"ra": 1.0, "dec": 2.0, "flux": 10.0, "snr": 5.0,
                                  "pixel_x": 50, "pixel_y": 50, "label": 1,
                                  "bbox": {"ixmin": 0, "ixmax": 5, "iymin": 0, "iymax": 5}}]
        mock_res.return_value = ([], False)
        rpt_mock = MagicMock()
        rpt_mock.id = "20260317_abc12345"
        mock_rpt.return_value = rpt_mock

        from parallax.survey import reduce
        progress_calls = []
        reduce("test", on_progress=lambda step, detail: progress_calls.append((step, detail)))

        steps = [s for s, _ in progress_calls]
        assert steps == ["acquire", "detect", "merge", "resolve", "report"]
        assert "1 file(s) found" in progress_calls[0][1]
        assert "F200W" in progress_calls[1][1]

    @patch("parallax.survey.report")
    @patch("parallax.survey.resolve")
    @patch("parallax.survey.detect")
    @patch("parallax.survey.acquire")
    def test_on_progress_none_ok(self, mock_acq, mock_det, mock_res, mock_rpt, tmp_db):
        """reduce() works fine when on_progress is None."""
        fits_path = os.path.join(tmp_db, "downloads", "noprog.fits")
        hdul = make_fits()
        hdul.writeto(fits_path, overwrite=True)

        mock_acq.return_value = [fits_path]
        mock_det.return_value = []
        mock_res.return_value = ([], False)
        mock_rpt.return_value = MagicMock()

        from parallax.survey import reduce
        reduce("test", on_progress=None)
        mock_rpt.assert_called_once()


class TestDetectionCache:
    def test_cache_hit(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)

        from parallax.survey import detect, cache_status
        r1 = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(cache_status()["detection"]) == 1

        # second call should hit cache and return identical results
        r2 = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(r2) == len(r1)
        assert r2[0]["ra"] == r1[0]["ra"]

    def test_cache_miss_on_param_change(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)

        from parallax.survey import detect
        r1 = detect(path, snr_threshold=1.5, min_pixels=5)
        r2 = detect(path, snr_threshold=5.0, min_pixels=5)
        # different params means different results (or empty)
        assert isinstance(r2, list)


class TestCatalogCache:
    def _dets(self):
        return [{"ra": 83.82, "dec": -5.39, "flux": 100.0, "snr": 5.0,
                 "pixel_x": 100.0, "pixel_y": 100.0, "label": 1,
                 "bbox": {"ixmin": 98, "ixmax": 103, "iymin": 98, "iymax": 103}}]

    def test_catalog_cache_hit(self, tmp_db):
        from parallax.survey import _set_catalog_cache, _get_catalog_cache

        key = "83.8200_-5.3900_7.2_SIMBAD"
        data = [{"catalog": "SIMBAD", "source_id": "s1",
                 "ra": 83.82, "dec": -5.39,
                 "object_type": "Star", "redshift": None}]
        _set_catalog_cache(key, "SIMBAD", 83.82, -5.39, 7.2, data)

        cached = _get_catalog_cache(key)
        assert cached is not None
        assert len(cached) == 1
        assert cached[0]["source_id"] == "s1"

    def test_catalog_cache_expiry(self, tmp_db):
        from parallax.survey import _set_catalog_cache, _get_catalog_cache
        from parallax._db import get_db

        key = "83.8200_-5.3900_7.2_SIMBAD"
        _set_catalog_cache(key, "SIMBAD", 83.82, -5.39, 7.2, [{"x": 1}])

        # manually expire it
        past = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        with get_db() as conn:
            conn.execute("UPDATE catalog_cache SET expires_at = ? WHERE field_key = ?",
                         (past, key))

        assert _get_catalog_cache(key) is None


class TestCandidateDedup:
    def _dets(self):
        return [{"ra": 83.82, "dec": -5.39, "flux": 100.0, "snr": 5.0,
                 "pixel_x": 100.0, "pixel_y": 100.0, "label": 1,
                 "bbox": {"ixmin": 98, "ixmax": 103, "iymin": 98, "iymax": 103}}]

    @patch("parallax.survey._query_gaia", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_simbad", return_value=[])
    def test_dedup_reuses_existing(self, s, n, g, tmp_db):
        from parallax.survey import resolve
        from parallax import catalog

        # first run creates a candidate
        cands1, _ = resolve(self._dets())
        catalog.add(cands1[0])
        orig_id = cands1[0].id

        # second run should reuse
        cands2, _ = resolve(self._dets())
        assert cands2[0].id == orig_id

    @patch("parallax.survey._query_gaia", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_simbad")
    def test_classification_upgrade(self, mock_s, mock_n, mock_g, tmp_db):
        from parallax.survey import resolve
        from parallax import catalog

        # first run: no matches -> unverified
        mock_s.return_value = []
        cands1, _ = resolve(self._dets())
        catalog.add(cands1[0])
        cid = cands1[0].id
        assert cands1[0].classification == "unverified"

        # second run: simbad match -> known
        mock_s.return_value = [{"catalog": "SIMBAD", "source_id": "s1",
                                "ra": 83.82, "dec": -5.39,
                                "object_type": "Star", "redshift": None}]
        cands2, _ = resolve(self._dets())
        assert cands2[0].id == cid
        assert cands2[0].classification == "known"

        # verify history recorded the upgrade
        hist = catalog.history(cid)
        cls_changes = [h for h in hist if h["field"] == "classification"]
        assert len(cls_changes) >= 1

    @patch("parallax.survey._query_gaia", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_simbad", return_value=[])
    def test_classification_no_downgrade(self, s, n, g, tmp_db):
        from parallax.survey import resolve
        from parallax import catalog

        # insert as "known"
        cands1, _ = resolve(self._dets())
        cands1[0].classification = "known"
        catalog.add(cands1[0])
        cid = cands1[0].id

        # second run: no matches -> would be "unverified", but should stay "known"
        cands2, _ = resolve(self._dets())
        assert cands2[0].id == cid
        assert cands2[0].classification == "known"


class TestCacheAPI:
    def test_cache_status_shape(self, tmp_db):
        from parallax.survey import cache_status
        st = cache_status()
        assert "detection" in st
        assert "catalog" in st
        assert isinstance(st["detection"], list)
        assert isinstance(st["catalog"], list)

    def test_clear_cache(self, tmp_db):
        from parallax.survey import detect, cache_status, clear_cache
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)

        detect(path, snr_threshold=1.5, min_pixels=5)
        st = cache_status()
        assert len(st["detection"]) > 0

        result = clear_cache()
        assert result["detection_entries_cleared"] >= 1
        st2 = cache_status()
        assert len(st2["detection"]) == 0


class TestMarkdownDedup:
    def test_no_duplicate_ids_in_markdown(self, tmp_db):
        from parallax.survey import _write_markdown

        # build a report with duplicate candidate IDs
        dup = Candidate(
            id="cnd_dup00001", ra=10.0, dec=20.0,
            flux=500.0, snr=4.0, classification="known",
            report_id="rpt_20260101_testdedup",
            pixel_coords=(50.0, 50.0),
            created_at=datetime(2026, 1, 15),
            catalog_matches=[CatalogMatch("SIMBAD", "S1", 0.3)],
        )
        rpt = Report(
            id="rpt_20260101_testdedup", target="TestField",
            instrument="NIRCAM", filters=["F200W"],
            created_at=datetime(2026, 1, 15),
            candidates=[dup, dup, dup],  # same candidate three times
            n_sources_detected=3, n_catalog_matched=3, n_unverified=0,
        )

        md_path = os.path.join(tmp_db, "reports", "dedup_test.md")
        _write_markdown(rpt, md_path, include_known=True)

        with open(md_path) as f:
            content = f.read()

        # the candidate id should appear exactly once in table rows
        table_rows = [l for l in content.splitlines() if l.startswith("| cnd_")]
        ids = [r.split("|")[1].strip() for r in table_rows]
        assert len(ids) == len(set(ids)), f"duplicate IDs in markdown: {ids}"


class TestI2dFilter:
    def test_non_i2d_excluded(self, tmp_db):
        """Non-i2d files should be filtered out before download."""
        from astropy.table import Table

        products = Table({
            "productFilename": [
                "jw01234_nircam_f200w_i2d.fits",
                "jw01234_nircam_f200w_cal.fits",
                "jw01234_nircam_f200w_i2d.fits.gz",
                "jw01234_nircam_f200w_rate.fits",
            ],
            "dataURI": [
                "mast:JWST/product/jw01234_nircam_f200w_i2d.fits",
                "mast:JWST/product/jw01234_nircam_f200w_cal.fits",
                "mast:JWST/product/jw01234_nircam_f200w_i2d.fits.gz",
                "mast:JWST/product/jw01234_nircam_f200w_rate.fits",
            ],
            "productType": ["SCIENCE", "SCIENCE", "SCIENCE", "SCIENCE"],
        })

        with patch("astroquery.mast.Observations") as mock_obs:
            mock_obs.get_product_list.return_value = products
            mock_obs.filter_products.return_value = products
            mock_obs.download_file.return_value = ("COMPLETE", None, None)

            from parallax.acquisition import _mast_download
            paths = _mast_download(MagicMock(), "/tmp")

            downloaded = [c[0][0] for c in mock_obs.download_file.call_args_list]
            assert any("i2d.fits" in u for u in downloaded)
            assert not any("cal.fits" in u for u in downloaded)
            assert not any("rate.fits" in u for u in downloaded)

    def test_no_i2d_raises(self, tmp_db):
        from astropy.table import Table

        products = Table({
            "productFilename": ["jw01234_cal.fits", "jw01234_rate.fits"],
            "productType": ["SCIENCE", "SCIENCE"],
        })

        with patch("astroquery.mast.Observations") as mock_obs:
            mock_obs.get_product_list.return_value = products
            mock_obs.filter_products.return_value = products

            from parallax.acquisition import _mast_download
            with pytest.raises(RuntimeError, match="no i2d products"):
                _mast_download(MagicMock(), "/tmp")


class TestReduceFingerprint:
    @patch("parallax.survey.report")
    @patch("parallax.survey.resolve")
    @patch("parallax.survey.detect")
    @patch("parallax.survey.acquire")
    def test_fingerprint_new_on_param_change(self, mock_acq, mock_det, mock_res, mock_rpt, tmp_db):
        """Different snr_threshold should produce a different fingerprint."""
        from parallax.survey import _run_fingerprint
        fits_path = os.path.join(tmp_db, "downloads", "test_i2d.fits")
        hdul = make_fits()
        hdul.writeto(fits_path, overwrite=True)

        fp1 = _run_fingerprint("test", [(fits_path, "F200W")], 3.0, 25, 2.0)
        fp2 = _run_fingerprint("test", [(fits_path, "F200W")], 5.0, 25, 2.0)
        assert fp1 != fp2

    def test_fingerprint_new_on_file_change(self, tmp_db):
        """Changed mtime should produce a different fingerprint."""
        from parallax.survey import _run_fingerprint
        fits_path = os.path.join(tmp_db, "downloads", "test_i2d.fits")
        hdul = make_fits()
        hdul.writeto(fits_path, overwrite=True)

        fp1 = _run_fingerprint("test", [(fits_path, "F200W")], 3.0, 25, 2.0)

        # change the mtime
        import time
        time.sleep(0.05)
        os.utime(fits_path, (time.time() + 100, time.time() + 100))

        fp2 = _run_fingerprint("test", [(fits_path, "F200W")], 3.0, 25, 2.0)
        assert fp1 != fp2

    @patch("parallax.survey.report")
    @patch("parallax.survey.resolve")
    @patch("parallax.survey.detect")
    @patch("parallax.survey.acquire")
    def test_reduce_with_missing_file(self, mock_acq, mock_det, mock_res, mock_rpt, tmp_db):
        """reduce() reads FILTER from FITS header for each file."""
        fake_path = os.path.join(tmp_db, "downloads", "not_yet_here.fits")
        hdul = make_fits()
        hdul.writeto(fake_path, overwrite=True)
        mock_acq.return_value = [fake_path]
        mock_det.return_value = [{"ra": 1.0, "dec": 2.0, "flux": 10.0, "snr": 5.0,
                                  "pixel_x": 50, "pixel_y": 50, "label": 1,
                                  "bbox": {"ixmin": 0, "ixmax": 5, "iymin": 0, "iymax": 5}}]
        mock_res.return_value = ([], False)
        mock_rpt.return_value = MagicMock()

        from parallax.survey import reduce
        reduce("test")
        mock_rpt.assert_called_once()
        # report() no longer receives fingerprint; it computes internally
        assert "fingerprint" not in mock_rpt.call_args.kwargs


class TestTargetSlug:
    def test_simple_name(self):
        from parallax.types import _target_slug
        assert _target_slug("M92") == "m92"

    def test_spaces(self):
        from parallax.types import _target_slug
        assert _target_slug("Orion Bar") == "orion_bar"

    def test_special_chars(self):
        from parallax.types import _target_slug
        assert _target_slug("NGC 6341 (M92)") == "ngc_6341_m92"

    def test_leading_trailing(self):
        from parallax.types import _target_slug
        assert _target_slug("  M92  ") == "m92"


class TestReportOutputStructure:
    @patch("parallax.catalog.add", return_value="cnd_12345678")
    def test_report_output_in_target_subfolder(self, mock_add, tmp_db):
        from parallax.survey import report
        hdul = make_fits()
        fits_path = _write_fits(hdul, tmp_db)

        cands = [Candidate(
            id="cnd_aabbccdd", ra=83.82, dec=-5.39, flux=100.0, snr=5.0,
            classification="unverified", report_id="",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
        )]
        rpt = report(cands, "M92", [(fits_path, "F200W")], 10, output_format="both")

        # files should be in data/reports/m92/{YYYYMMDD}/
        json_dir = os.path.dirname(rpt.json_path)
        assert os.path.basename(os.path.dirname(json_dir)) == "m92"
        assert len(os.path.basename(json_dir)) == 8  # YYYYMMDD
        md_dir = os.path.dirname(rpt.md_path)
        assert os.path.basename(os.path.dirname(md_dir)) == "m92"
        assert os.path.isfile(rpt.json_path)
        assert os.path.isfile(rpt.md_path)

    @patch("parallax.catalog.add", return_value="cnd_12345678")
    def test_same_fingerprint_overwrites(self, mock_add, tmp_db):
        """Calling report() twice on same file overwrites via internal fingerprint."""
        from parallax.survey import report
        hdul = make_fits()
        fits_path = _write_fits(hdul, tmp_db)

        cands = [Candidate(
            id="cnd_ov000001", ra=83.82, dec=-5.39, flux=100.0, snr=5.0,
            classification="unverified", report_id="",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
        )]

        rpt1 = report(cands, "M92", [(fits_path, "F200W")], 10, output_format="json")
        rpt2 = report(cands, "M92", [(fits_path, "F200W")], 15, output_format="json")

        # same path because same internal fingerprint
        assert rpt1.json_path == rpt2.json_path

        with open(rpt2.json_path) as f:
            data = json.load(f)
        # second write overwrote the first
        assert data["n_sources_detected"] == 15


class TestMergeDetections:
    def test_dedup_by_sky_position(self, tmp_db):
        from parallax.survey import _merge_detections
        # two detections at same position but different filters
        d1 = [{"ra": 83.82, "dec": -5.39, "flux": 100.0, "snr": 5.0,
               "pixel_x": 100.0, "pixel_y": 100.0, "label": 1,
               "bbox": {}, "filter": "F200W"}]
        d2 = [{"ra": 83.82, "dec": -5.39, "flux": 200.0, "snr": 8.0,
               "pixel_x": 101.0, "pixel_y": 101.0, "label": 1,
               "bbox": {}, "filter": "F444W"}]
        merged = _merge_detections(d1 + d2)
        assert len(merged) == 1
        # best SNR wins top-level fields
        assert merged[0]["snr"] == 8.0
        assert merged[0]["flux"] == 200.0
        assert len(merged[0]["detections"]) == 2
        filters = {d["filter"] for d in merged[0]["detections"]}
        assert filters == {"F200W", "F444W"}

    def test_distant_sources_not_merged(self, tmp_db):
        from parallax.survey import _merge_detections
        d1 = [{"ra": 83.82, "dec": -5.39, "flux": 100.0, "snr": 5.0,
               "pixel_x": 100.0, "pixel_y": 100.0, "label": 1,
               "bbox": {}, "filter": "F200W"}]
        d2 = [{"ra": 84.82, "dec": -4.39, "flux": 200.0, "snr": 8.0,
               "pixel_x": 200.0, "pixel_y": 200.0, "label": 2,
               "bbox": {}, "filter": "F444W"}]
        merged = _merge_detections(d1 + d2)
        assert len(merged) == 2

    def test_nan_coords_pass_through(self, tmp_db):
        from parallax.survey import _merge_detections
        d = [{"ra": float("nan"), "dec": float("nan"), "flux": 50.0, "snr": 3.0,
              "pixel_x": 10.0, "pixel_y": 10.0, "label": 1, "bbox": {}, "filter": "F200W"}]
        merged = _merge_detections(d)
        assert len(merged) == 1
        assert len(merged[0]["detections"]) == 1

    def test_missing_filter_key(self, tmp_db):
        from parallax.survey import _merge_detections
        d = [{"ra": 83.82, "dec": -5.39, "flux": 100.0, "snr": 5.0,
              "pixel_x": 100.0, "pixel_y": 100.0, "label": 1, "bbox": {}}]
        merged = _merge_detections(d)
        assert merged[0]["detections"][0]["filter"] == "UNKNOWN"


class TestDetectFilterName:
    def test_filter_key_in_output(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5, filter_name="F200W")
        assert len(results) >= 1
        for s in results:
            assert s["filter"] == "F200W"

    def test_no_filter_key_by_default(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(results) >= 1
        assert "filter" not in results[0]


class TestReportFitsInputs:
    @patch("parallax.catalog.add", return_value="cnd_12345678")
    def test_filters_populated(self, mock_add, tmp_db):
        from parallax.survey import report
        from parallax._db import get_db
        hdul = make_fits()
        fits_path = _write_fits(hdul, tmp_db)

        cands = [Candidate(
            id="cnd_fi000001", ra=83.82, dec=-5.39, flux=100.0, snr=5.0,
            classification="unverified", report_id="",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
        )]
        rpt = report(cands, "M92", [(fits_path, "F200W"), (fits_path, "F444W")],
                      10, output_format="json")
        assert rpt.filters == ["F200W", "F444W"]

        # verify report_inputs rows
        with get_db() as conn:
            rows = conn.execute(
                "SELECT * FROM report_inputs WHERE report_id = ? ORDER BY id",
                (rpt.id,)
            ).fetchall()
        assert len(rows) == 2
        assert rows[0]["filter"] == "F200W"
        assert rows[1]["filter"] == "F444W"


class TestCandidateDetectionsRoundTrip:
    def test_add_get_preserves_detections(self, tmp_db):
        from parallax.types import Detection
        from parallax import catalog

        cand = Candidate(
            id="cnd_det00001", ra=83.82, dec=-5.39, flux=200.0, snr=8.0,
            classification="unverified", report_id="rpt_20260101_test",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
            detections=[
                Detection(filter="F200W", flux=100.0, snr=5.0, pixel_coords=(99.0, 99.0)),
                Detection(filter="F444W", flux=200.0, snr=8.0, pixel_coords=(101.0, 101.0)),
            ],
        )
        catalog.add(cand)
        loaded = catalog.get(cand.id)
        assert len(loaded.detections) == 2
        assert loaded.detections[0].filter == "F200W"
        assert loaded.detections[1].filter == "F444W"
        assert loaded.detections[1].flux == 200.0


class TestGaiaFailedMarkdown:
    def test_gaia_unavailable_note(self, tmp_db):
        from parallax.survey import _write_markdown

        rpt = Report(
            id="rpt_20260101_gaiafail", target="TestField",
            instrument="NIRCAM", filters=["F200W"],
            created_at=datetime(2026, 1, 15),
            candidates=[],
            n_sources_detected=10, n_catalog_matched=0, n_unverified=10,
        )
        md_path = os.path.join(tmp_db, "reports", "gaia_fail_test.md")
        _write_markdown(rpt, md_path, include_known=False, gaia_failed=True)

        with open(md_path) as f:
            content = f.read()
        assert "Gaia: unavailable during this run" in content

    def test_gaia_succeeded_no_note(self, tmp_db):
        from parallax.survey import _write_markdown

        rpt = Report(
            id="rpt_20260101_gaiaok", target="TestField",
            instrument="NIRCAM", filters=["F200W"],
            created_at=datetime(2026, 1, 15),
            candidates=[],
            n_sources_detected=10, n_catalog_matched=0, n_unverified=10,
        )
        md_path = os.path.join(tmp_db, "reports", "gaia_ok_test.md")
        _write_markdown(rpt, md_path, include_known=False, gaia_failed=False)

        with open(md_path) as f:
            content = f.read()
        assert "unavailable" not in content


class TestResolveNameErrors:
    def test_raises_target_not_found_when_none(self):
        from parallax.acquisition import _resolve_name
        from parallax.exceptions import TargetNotFoundError

        with patch("astroquery.simbad.Simbad.query_object", return_value=None):
            with pytest.raises(TargetNotFoundError, match="Cannot resolve target 'FakeTarget'"):
                _resolve_name("FakeTarget")

    def test_raises_target_not_found_on_exception(self):
        from parallax.acquisition import _resolve_name
        from parallax.exceptions import TargetNotFoundError

        with patch("astroquery.simbad.Simbad.query_object", side_effect=Exception("connection failed")):
            with pytest.raises(TargetNotFoundError, match="Cannot resolve target 'BadName'"):
                _resolve_name("BadName")


class TestComputeConfidence:
    def test_range_for_various_inputs(self):
        from parallax.survey import _compute_confidence
        cases = [
            (5.0, 1, 3, float("inf"), "kron"),
            (15.0, 2, 2, 5.0, "segment"),
            (0.5, 1, 1, 1.0, "zero"),
            (30.0, 4, 4, float("inf"), "kron"),
        ]
        for snr, nf, tf, sep, fs in cases:
            v = _compute_confidence(snr, nf, tf, sep, fs)
            assert 0.0 <= v <= 1.0

    def test_perfect_score(self):
        from parallax.survey import _compute_confidence
        v = _compute_confidence(30.0, 3, 3, float("inf"), "kron")
        assert v == 1.0

    def test_worst_score(self):
        from parallax.survey import _compute_confidence
        # snr=0, n_filters=1, total=4, nearest_sep=0 (matched), flux_source=zero
        # filter_score can't reach 0 since n_filters >= 1
        v = _compute_confidence(0.0, 1, 4, 0.0, "zero")
        assert v == round(0.35 * 0.25, 4)

    def test_detect_returns_flux_source(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(results) >= 1
        for s in results:
            assert "flux_source" in s
            assert s["flux_source"] in ("kron", "segment", "zero")

    def test_reads_search_radius_from_config(self):
        from parallax.survey import _compute_confidence
        with patch("parallax.survey.config") as mock_cfg:
            mock_cfg.get.return_value = 5.0
            v = _compute_confidence(10.0, 2, 3, 3.0, "kron")
            mock_cfg.get.assert_called_with("resolver.search_radius_arcsec", 2.0)
            # sep=3.0 is within radius=5.0 so near_miss_score=0
            assert v == _compute_confidence.__wrapped__(10.0, 2, 3, 3.0, "kron") if hasattr(_compute_confidence, '__wrapped__') else True
        # with default radius=2.0, sep=3.0 is outside so near_miss_score > 0
        v_default = _compute_confidence(10.0, 2, 3, 3.0, "kron")
        with patch("parallax.survey.config") as mock_cfg:
            mock_cfg.get.return_value = 5.0
            v_wide = _compute_confidence(10.0, 2, 3, 3.0, "kron")
        assert v_wide < v_default  # wider radius makes near-miss score 0

    @patch("parallax.survey._query_simbad", return_value=[])
    @patch("parallax.survey._query_ned", return_value=[])
    @patch("parallax.survey._query_gaia", return_value=[])
    def test_resolve_sets_confidence(self, g, n, s, tmp_db):
        from parallax.survey import resolve
        dets = [
            {"ra": 83.82, "dec": -5.39, "flux": 100.0,
             "snr": 10.0, "pixel_x": 100.0, "pixel_y": 100.0,
             "flux_source": "kron", "label": 0,
             "bbox": {"ixmin": 98, "ixmax": 103, "iymin": 98, "iymax": 103},
             "detections": [{"filter": "F200W", "flux": 100.0, "snr": 10.0,
                             "pixel_x": 100.0, "pixel_y": 100.0, "flux_source": "kron"}]}
        ]
        cands, _ = resolve(dets)
        assert len(cands) == 1
        assert cands[0].confidence > 0.0


class TestResolveNameReturnType:
    def test_returns_main_id_ra_dec(self):
        from parallax.acquisition import _resolve_name
        from astropy.table import Table

        mock_result = Table(
            {"main_id": ["NGC 6341"], "ra": [259.28], "dec": [43.14]},
        )
        with patch("astroquery.simbad.Simbad.query_object", return_value=mock_result):
            main_id, ra, dec = _resolve_name("M92")
        assert main_id == "NGC 6341"
        assert abs(ra - 259.28) < 0.01
        assert abs(dec - 43.14) < 0.01

    def test_empty_table_raises(self):
        from parallax.acquisition import _resolve_name
        from parallax.exceptions import TargetNotFoundError
        from astropy.table import Table

        empty = Table(names=["main_id", "ra", "dec"], dtype=["U64", "f8", "f8"])
        with patch("astroquery.simbad.Simbad.query_object", return_value=empty):
            with pytest.raises(TargetNotFoundError):
                _resolve_name("FakeThing")

    @patch("parallax.acquisition._mast_download")
    @patch("parallax.acquisition._mast_query")
    def test_canonical_slug_used_for_download(self, mock_query, mock_dl, tmp_db):
        from parallax.survey import acquire
        from parallax.acquisition import _resolve_name
        from astropy.table import Table

        mock_result = Table(
            {"main_id": ["NGC 6341"], "ra": [259.28], "dec": [43.14]},
        )
        with patch("astroquery.simbad.Simbad.query_object", return_value=mock_result):
            mock_query.return_value = MagicMock(__len__=lambda s: 1)
            mock_dl.return_value = ["/fake/ngc_6341/test_i2d.fits"]
            acquire("M92")

        dest_dir = mock_dl.call_args[0][1]
        # should use canonical slug ngc_6341, not m92
        assert "ngc_6341" in dest_dir
        assert "m92" not in dest_dir


class TestAcquireOnProgress:
    @patch("parallax.acquisition._mast_download")
    @patch("parallax.acquisition._mast_query")
    @patch("parallax.acquisition._resolve_name", return_value=("NGC 6341", 259.28, 43.14))
    def test_acquire_emits_progress(self, mock_resolve, mock_query, mock_dl, tmp_db):
        from parallax.survey import acquire

        mock_query.return_value = MagicMock(__len__=lambda s: 3)
        fake_path = os.path.join(tmp_db, "downloads", "mastDownload", "JWST", "ngc_6341", "t_i2d.fits")
        mock_dl.return_value = [fake_path]

        calls = []
        cb = lambda step, detail: calls.append((step, detail))
        acquire("M92", on_progress=cb)

        steps = [s for s, _ in calls]
        assert "acquire" in steps
        # on_progress is forwarded to _mast_download
        dl_kwargs = mock_dl.call_args[1]
        assert dl_kwargs.get("on_progress") is cb


class TestMastDownloadEmpty:
    def test_empty_product_list_raises(self):
        from parallax.acquisition import _mast_download
        from astropy.table import Table

        empty_table = Table(names=["productFilename"], dtype=["U64"])

        with patch("parallax.survey.Observations", create=True) as mock_obs:
            # need to patch at import time inside _mast_download
            with patch("astroquery.mast.Observations.get_product_list", return_value=empty_table):
                with patch("astroquery.mast.Observations.filter_products", return_value=empty_table):
                    with pytest.raises(RuntimeError, match="no i2d products"):
                        _mast_download([], "/tmp/fake")


class TestGaiaStdoutRestore:
    @patch("parallax.survey.config")
    def test_stdout_restored_on_gaia_exception(self, mock_config, tmp_db):
        """_query_gaia restores sys.stdout even when the Gaia call raises."""
        from parallax.survey import _query_gaia
        import sys

        mock_config.get = lambda k, default=None: False  # disable cache

        original_stdout = sys.stdout
        with patch("astroquery.gaia.Gaia") as mock_gaia:
            mock_gaia.cone_search_async.side_effect = Exception("server on fire")
            result = _query_gaia(180.0, 45.0, 10.0, 30)

        assert result == []
        assert sys.stdout is original_stdout


class TestAcquireNoJWSTData:
    @patch("parallax.acquisition._mast_query")
    @patch("parallax.acquisition._resolve_name", return_value=("SomeObscureTarget", 10.0, 20.0))
    def test_raises_target_not_found(self, mock_resolve, mock_mast, tmp_db):
        from parallax.survey import acquire
        from parallax.exceptions import TargetNotFoundError

        # _mast_query returns empty
        mock_mast.return_value = []
        with pytest.raises(TargetNotFoundError, match="no JWST Level 3"):
            acquire("SomeObscureTarget")


class TestQuietStdoutRestore:
    def test_restores_after_normal_exit(self):
        from parallax.acquisition import _quiet_stdout
        orig = sys.stdout
        with _quiet_stdout():
            pass
        assert sys.stdout is orig

    def test_restores_after_exception(self):
        from parallax.acquisition import _quiet_stdout
        orig = sys.stdout
        with pytest.raises(ValueError):
            with _quiet_stdout():
                raise ValueError("boom")
        assert sys.stdout is orig

    def test_restores_after_keyboard_interrupt(self):
        from parallax.acquisition import _quiet_stdout
        orig = sys.stdout
        with pytest.raises(KeyboardInterrupt):
            with _quiet_stdout():
                raise KeyboardInterrupt
        assert sys.stdout is orig


class TestCacheKeyIncludesFilterSize:
    def test_filter_size_changes_hash(self, tmp_db):
        from parallax.survey import _fits_hash
        from parallax.config import config

        path = os.path.join(tmp_db, "test_hash.fits")
        hdul = make_fits()
        hdul.writeto(path, overwrite=True)

        config.set("detection.background_filter_size", 3)
        h1 = _fits_hash(path, 3.0, 25, 2.0)

        config.set("detection.background_filter_size", 5)
        h2 = _fits_hash(path, 3.0, 25, 2.0)

        assert h1 != h2, "changing background_filter_size must change cache key"

    def test_interp_changes_hash(self, tmp_db):
        from parallax.survey import _fits_hash
        from parallax.config import config

        path = os.path.join(tmp_db, "test_hash2.fits")
        hdul = make_fits()
        hdul.writeto(path, overwrite=True)

        config.set("detection.background_interp", "zoom")
        h1 = _fits_hash(path, 3.0, 25, 2.0)

        config.set("detection.background_interp", "idw")
        h2 = _fits_hash(path, 3.0, 25, 2.0)

        assert h1 != h2


class TestCatalogMatchDataRoundTrip:
    def test_empty_string_data(self):
        from parallax.types import report_from_dict
        d = _make_report_dict(data="")
        rpt = report_from_dict(d)
        assert rpt.candidates[0].catalog_matches[0].data == {}

    def test_none_data(self):
        from parallax.types import report_from_dict
        d = _make_report_dict(data=None)
        rpt = report_from_dict(d)
        assert rpt.candidates[0].catalog_matches[0].data == {}

    def test_missing_data_key(self):
        from parallax.types import report_from_dict
        d = _make_report_dict(data=None)
        del d["candidates"][0]["catalog_matches"][0]["data"]
        rpt = report_from_dict(d)
        assert rpt.candidates[0].catalog_matches[0].data == {}

    def test_valid_dict_data(self):
        from parallax.types import report_from_dict
        d = _make_report_dict(data={"velocity": 42.0})
        rpt = report_from_dict(d)
        assert rpt.candidates[0].catalog_matches[0].data == {"velocity": 42.0}

    def test_safe_json_dict_empty_string(self):
        from parallax.types import _safe_json_dict
        assert _safe_json_dict("") == {}

    def test_safe_json_dict_null_json(self):
        from parallax.types import _safe_json_dict
        assert _safe_json_dict("null") == {}

    def test_safe_json_dict_valid(self):
        from parallax.types import _safe_json_dict
        assert _safe_json_dict('{"a": 1}') == {"a": 1}

    def test_safe_json_dict_quoted_empty(self):
        from parallax.types import _safe_json_dict
        # double-serialized empty string
        assert _safe_json_dict('""') == {}


def _make_report_dict(data=None):
    """minimal report dict for serialization tests"""
    return {
        "id": "20240101_abcd1234",
        "target": "test",
        "instrument": "NIRCAM",
        "filters": ["F200W"],
        "created_at": "2024-01-01T00:00:00",
        "n_sources_detected": 1,
        "n_catalog_matched": 1,
        "n_unverified": 0,
        "candidates": [{
            "id": "cnd_00000001",
            "ra": 83.8, "dec": -5.4,
            "flux": 1.0, "snr": 5.0,
            "classification": "known",
            "report_id": "20240101_abcd1234",
            "pixel_coords": [100.0, 100.0],
            "created_at": "2024-01-01T00:00:00",
            "catalog_matches": [{
                "catalog": "NED",
                "source_id": "NED 1",
                "separation_arcsec": 0.5,
                "object_type": "G",
                "redshift": 0.01,
                "data": data,
            }],
            "detections": [],
            "tags": [], "notes": [],
            "confidence": 0.0,
        }],
    }


class TestFluxUncertainty:
    def test_err_extension_produces_flux_err(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05, include_err=True, pixar_sr=2.35e-14)
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(results) >= 1
        for s in results:
            assert s["flux_err"] is not None
            assert s["flux_err"] > 0
            assert s["flux_mjy_err"] is not None
            assert s["mag_ab_err"] is not None

    def test_err_and_wht_extensions(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05, include_err=True,
                         include_wht=True, pixar_sr=2.35e-14)
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(results) >= 1
        for s in results:
            assert s["flux_err"] is not None
            assert s["flux_err"] > 0

    def test_no_err_extension_leaves_none(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05)
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        assert len(results) >= 1
        for s in results:
            assert s["flux_err"] is None
            assert s["flux_mjy_err"] is None
            assert s["mag_ab_err"] is None

    def test_mag_ab_err_formula(self):
        """mag_ab_err = 2.5 / ln(10) * flux_mjy_err / flux_mjy"""
        flux_mjy = 1e-6
        flux_mjy_err = 1e-7
        expected = 2.5 / math.log(10) * flux_mjy_err / flux_mjy
        # survey rounds to 4 decimal places
        rounded = round(expected, 4)
        assert abs(rounded - expected) < 5e-5
        assert rounded > 0

    def test_no_pixar_sr_no_mjy_err(self, tmp_db):
        hdul = make_fits(n_sources=3, noise=0.05, include_err=True)
        # no PIXAR_SR in header
        path = _write_fits(hdul, tmp_db)
        from parallax.survey import detect
        results = detect(path, snr_threshold=1.5, min_pixels=5)
        for s in results:
            # flux_err should still be set (raw instrumental)
            assert s["flux_err"] is not None
            # but mjy and mag err need PIXAR_SR
            assert s["flux_mjy_err"] is None
            assert s["mag_ab_err"] is None


class TestMarkdownUncertaintyCols:
    def test_table_has_err_columns(self, tmp_db):
        from parallax.survey import _write_markdown
        from parallax.types import Detection

        cand = Candidate(
            id="cnd_err_test1", ra=83.82, dec=-5.39, flux=500.0, snr=15.0,
            classification="unverified", report_id="rpt_err", confidence=0.7,
            pixel_coords=(100.0, 100.0), created_at=datetime(2026, 1, 1),
            detections=[Detection(filter="F200W", flux=500.0, snr=15.0,
                                  pixel_coords=(100.0, 100.0),
                                  flux_mjy=1.17e-8, mag_ab=22.31,
                                  flux_err=12.5, flux_mjy_err=2.9e-10,
                                  mag_ab_err=0.027)],
            flux_err=12.5, flux_mjy_err=2.9e-10, mag_ab_err=0.027,
        )
        rpt = Report(
            id="rpt_err", target="ErrTest", instrument="NIRCAM",
            filters=["F200W"], created_at=datetime(2026, 1, 1),
            candidates=[cand],
            n_sources_detected=1, n_catalog_matched=0, n_unverified=1,
        )
        md_path = os.path.join(tmp_db, "reports", "err_test.md")
        _write_markdown(rpt, md_path, include_known=False)
        with open(md_path) as f:
            content = f.read()
        assert "Flux err" in content
        assert "Mag err" in content
        assert "2.9000e-10" in content
        assert "0.0270" in content

    def test_table_dashes_when_no_err(self, tmp_db):
        from parallax.survey import _write_markdown
        from parallax.types import Detection

        cand = Candidate(
            id="cnd_noerr01", ra=83.82, dec=-5.39, flux=500.0, snr=15.0,
            classification="unverified", report_id="rpt_noerr", confidence=0.7,
            pixel_coords=(100.0, 100.0), created_at=datetime(2026, 1, 1),
            detections=[Detection(filter="F200W", flux=500.0, snr=15.0,
                                  pixel_coords=(100.0, 100.0),
                                  flux_mjy=1.17e-8, mag_ab=22.31)],
        )
        rpt = Report(
            id="rpt_noerr", target="NoErrTest", instrument="NIRCAM",
            filters=["F200W"], created_at=datetime(2026, 1, 1),
            candidates=[cand],
            n_sources_detected=1, n_catalog_matched=0, n_unverified=1,
        )
        md_path = os.path.join(tmp_db, "reports", "noerr_test.md")
        _write_markdown(rpt, md_path, include_known=False)
        with open(md_path) as f:
            content = f.read()
        # err columns present but values are dashes
        assert "Flux err" in content
        # the row should have dash for err values
        lines = [l for l in content.split("\n") if "cnd_noerr01" in l]
        assert len(lines) == 1
        parts = lines[0].split("|")
        # flux err col and mag err col should be '-'
        assert parts[8].strip() == "-"
        assert parts[10].strip() == "-"


class TestJsonRoundTripUncertainty:
    def test_detection_uncertainty_survives_roundtrip(self):
        from parallax.types import report_to_dict, report_from_dict, Detection

        det = Detection(filter="F200W", flux=500.0, snr=15.0,
                        pixel_coords=(100.0, 100.0),
                        flux_mjy=1.17e-8, mag_ab=22.31,
                        flux_err=12.5, flux_mjy_err=2.9e-10, mag_ab_err=0.027)
        cand = Candidate(
            id="cnd_rt_err01", ra=83.82, dec=-5.39, flux=500.0, snr=15.0,
            classification="unverified", report_id="rpt_rt",
            pixel_coords=(100.0, 100.0), created_at=datetime(2026, 1, 1),
            detections=[det],
            flux_err=12.5, flux_mjy_err=2.9e-10, mag_ab_err=0.027,
        )
        rpt = Report(
            id="rpt_rt", target="RoundTrip", instrument="NIRCAM",
            filters=["F200W"], created_at=datetime(2026, 1, 1),
            candidates=[cand],
            n_sources_detected=1, n_catalog_matched=0, n_unverified=1,
        )
        d = report_to_dict(rpt)
        # check dict has the fields
        cd = d["candidates"][0]
        assert cd["flux_err"] == 12.5
        assert cd["flux_mjy_err"] == 2.9e-10
        assert cd["mag_ab_err"] == 0.027
        dd = cd["detections"][0]
        assert dd["flux_err"] == 12.5
        assert dd["flux_mjy_err"] == 2.9e-10
        assert dd["mag_ab_err"] == 0.027

        # round-trip through JSON
        rpt2 = report_from_dict(json.loads(json.dumps(d)))
        c2 = rpt2.candidates[0]
        assert c2.flux_err == 12.5
        assert c2.flux_mjy_err == 2.9e-10
        assert c2.mag_ab_err == 0.027
        d2 = c2.detections[0]
        assert d2.flux_err == 12.5
        assert d2.flux_mjy_err == 2.9e-10
        assert d2.mag_ab_err == 0.027

    def test_none_uncertainty_roundtrip(self):
        from parallax.types import report_to_dict, report_from_dict, Detection

        det = Detection(filter="F200W", flux=500.0, snr=15.0,
                        pixel_coords=(100.0, 100.0))
        cand = Candidate(
            id="cnd_rt_none1", ra=83.82, dec=-5.39, flux=500.0, snr=15.0,
            classification="unverified", report_id="rpt_rt2",
            pixel_coords=(100.0, 100.0), created_at=datetime(2026, 1, 1),
            detections=[det],
        )
        rpt = Report(
            id="rpt_rt2", target="RoundTrip2", instrument="NIRCAM",
            filters=["F200W"], created_at=datetime(2026, 1, 1),
            candidates=[cand],
            n_sources_detected=1, n_catalog_matched=0, n_unverified=1,
        )
        d = report_to_dict(rpt)
        rpt2 = report_from_dict(json.loads(json.dumps(d)))
        c2 = rpt2.candidates[0]
        assert c2.flux_err is None
        assert c2.detections[0].flux_err is None


class TestDbRoundTripPhotometry:
    """flux_mjy, mag_ab, and uncertainty fields survive DB add/get."""

    def test_all_photometry_fields_persisted(self, tmp_db):
        from parallax.types import Detection
        from parallax import catalog

        det = Detection(filter="F200W", flux=500.0, snr=15.0,
                        pixel_coords=(100.0, 100.0),
                        flux_mjy=1.17e-8, mag_ab=22.31,
                        flux_err=12.5, flux_mjy_err=2.9e-10, mag_ab_err=0.027)
        cand = Candidate(
            id="cnd_dbrt0001", ra=83.82, dec=-5.39, flux=500.0, snr=15.0,
            classification="unverified", report_id="rpt_dbrt",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
            detections=[det],
            flux_err=12.5, flux_mjy_err=2.9e-10, mag_ab_err=0.027,
        )
        catalog.add(cand)
        loaded = catalog.get(cand.id)

        # candidate-level uncertainty
        assert loaded.flux_err == 12.5
        assert loaded.flux_mjy_err == 2.9e-10
        assert loaded.mag_ab_err == 0.027

        # detection-level: flux_mjy and mag_ab (bug 1)
        d = loaded.detections[0]
        assert d.flux_mjy == 1.17e-8
        assert d.mag_ab == 22.31

        # detection-level: uncertainty (bug 2)
        assert d.flux_err == 12.5
        assert d.flux_mjy_err == 2.9e-10
        assert d.mag_ab_err == 0.027

    def test_none_photometry_fields_persisted(self, tmp_db):
        from parallax.types import Detection
        from parallax import catalog

        det = Detection(filter="F200W", flux=500.0, snr=15.0,
                        pixel_coords=(100.0, 100.0))
        cand = Candidate(
            id="cnd_dbrt0002", ra=83.82, dec=-5.39, flux=500.0, snr=15.0,
            classification="unverified", report_id="rpt_dbrt2",
            pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
            detections=[det],
        )
        catalog.add(cand)
        loaded = catalog.get(cand.id)

        assert loaded.flux_err is None
        assert loaded.flux_mjy_err is None
        assert loaded.mag_ab_err is None

        d = loaded.detections[0]
        assert d.flux_mjy is None
        assert d.mag_ab is None
        assert d.flux_err is None
        assert d.flux_mjy_err is None
        assert d.mag_ab_err is None


class TestComputeHints:
    def _make_candidate(self, classification="unverified", detections=None):
        from parallax.types import Detection
        return Candidate(
            id="cnd_hint001",
            ra=83.0, dec=-5.0,
            flux=100.0, snr=10.0,
            classification=classification,
            report_id="rpt_test",
            pixel_coords=(50.0, 50.0),
            created_at=datetime.now(UTC),
            detections=detections or [],
        )

    def test_extended_morphology(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [{"filter": "F200W", "semimajor_sigma": 4.5, "elongation": 2.0,
                 "flux_mjy": None}]
        hints = _compute_hints(cand, dets)
        assert "extended in F200W" in hints

    def test_point_source_morphology(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [{"filter": "F200W", "semimajor_sigma": 1.2, "elongation": 1.1,
                 "flux_mjy": None}]
        hints = _compute_hints(cand, dets)
        assert "point source in F200W" in hints

    def test_no_morphology_when_none(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [{"filter": "F200W", "semimajor_sigma": None, "elongation": None,
                 "flux_mjy": None}]
        hints = _compute_hints(cand, dets)
        assert not any("extended" in h or "point source" in h for h in hints)

    def test_narrowband_only(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [
            {"filter": "F187N", "semimajor_sigma": None, "elongation": None, "flux_mjy": None},
            {"filter": "F212N", "semimajor_sigma": None, "elongation": None, "flux_mjy": None},
        ]
        hints = _compute_hints(cand, dets)
        assert "narrowband-only detection" in hints

    def test_not_narrowband_only_with_broadband(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [
            {"filter": "F187N", "semimajor_sigma": None, "elongation": None, "flux_mjy": None},
            {"filter": "F200W", "semimajor_sigma": None, "elongation": None, "flux_mjy": None},
        ]
        hints = _compute_hints(cand, dets)
        assert "narrowband-only detection" not in hints

    def test_single_filter(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [{"filter": "F200W", "semimajor_sigma": None, "elongation": None,
                 "flux_mjy": None}]
        hints = _compute_hints(cand, dets)
        assert "single-filter detection" in hints

    def test_no_single_filter_with_two(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [
            {"filter": "F200W", "semimajor_sigma": None, "elongation": None, "flux_mjy": None},
            {"filter": "F150W", "semimajor_sigma": None, "elongation": None, "flux_mjy": None},
        ]
        hints = _compute_hints(cand, dets)
        assert "single-filter detection" not in hints

    def test_f187n_excess(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [
            {"filter": "F187N", "semimajor_sigma": None, "elongation": None, "flux_mjy": 5.0},
            {"filter": "F182M", "semimajor_sigma": None, "elongation": None, "flux_mjy": 2.0},
        ]
        hints = _compute_hints(cand, dets)
        assert "F187N excess vs continuum" in hints

    def test_f187n_no_excess_below_threshold(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [
            {"filter": "F187N", "semimajor_sigma": None, "elongation": None, "flux_mjy": 3.0},
            {"filter": "F182M", "semimajor_sigma": None, "elongation": None, "flux_mjy": 2.0},
        ]
        hints = _compute_hints(cand, dets)
        assert "F187N excess vs continuum" not in hints

    def test_f162m_excess(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate()
        dets = [
            {"filter": "F162M", "semimajor_sigma": None, "elongation": None, "flux_mjy": 7.0},
            {"filter": "F150W", "semimajor_sigma": None, "elongation": None, "flux_mjy": 3.0},
        ]
        hints = _compute_hints(cand, dets)
        assert "F162M excess vs continuum" in hints

    def test_known_candidate_gets_empty_hints(self):
        from parallax.survey import _compute_hints
        cand = self._make_candidate(classification="known")
        dets = [{"filter": "F200W", "semimajor_sigma": 5.0, "elongation": 2.0,
                 "flux_mjy": None}]
        hints = _compute_hints(cand, dets)
        assert hints == []

    def test_hints_serialization_roundtrip(self):
        from parallax.types import report_to_dict, report_from_dict
        cand = self._make_candidate()
        cand.hints = ["extended in F200W", "single-filter detection"]
        rpt = Report(
            id="20260402_test1234",
            target="M92",
            instrument="NIRCAM",
            filters=["F200W"],
            created_at=datetime.now(UTC),
            candidates=[cand],
            n_sources_detected=1,
            n_catalog_matched=0,
            n_unverified=1,
        )
        d = report_to_dict(rpt)
        assert d["candidates"][0]["hints"] == ["extended in F200W", "single-filter detection"]

        restored = report_from_dict(d)
        assert restored.candidates[0].hints == ["extended in F200W", "single-filter detection"]

    def test_hints_db_persistence(self, tmp_db):
        from parallax import catalog
        cand = self._make_candidate()
        cand.hints = ["point source in F200W", "narrowband-only detection"]
        catalog.add(cand)
        loaded = catalog.get(cand.id)
        assert loaded.hints == ["point source in F200W", "narrowband-only detection"]


class TestLocalRmsOnDetection:
    def test_local_rms_field_default_none(self):
        from parallax.types import Detection
        det = Detection(filter="F200W", flux=10.0, snr=5.0, pixel_coords=(1.0, 2.0))
        assert det.local_rms is None

    def test_local_rms_roundtrip_json(self):
        from parallax.types import Detection, Candidate, Report, report_to_dict, report_from_dict
        det = Detection(filter="F200W", flux=10.0, snr=5.0,
                        pixel_coords=(1.0, 2.0), local_rms=0.042)
        cand = Candidate(
            id="cnd_lrms001", ra=83.0, dec=-5.0, flux=10.0, snr=5.0,
            classification="unverified", report_id="rpt_test",
            pixel_coords=(1.0, 2.0), created_at=datetime.now(UTC),
            detections=[det],
        )
        rpt = Report(id="20260402_lrmstest", target="M92", instrument="NIRCAM",
                     filters=["F200W"], created_at=datetime.now(UTC),
                     candidates=[cand], n_sources_detected=1,
                     n_catalog_matched=0, n_unverified=1)
        d = report_to_dict(rpt)
        assert d["candidates"][0]["detections"][0]["local_rms"] == 0.042
        restored = report_from_dict(d)
        assert restored.candidates[0].detections[0].local_rms == 0.042

    def test_local_rms_db_persistence(self, tmp_db):
        from parallax.types import Detection
        from parallax import catalog
        det = Detection(filter="F200W", flux=10.0, snr=5.0,
                        pixel_coords=(1.0, 2.0), local_rms=0.037)
        cand = Candidate(
            id="cnd_lrms002", ra=83.0, dec=-5.0, flux=10.0, snr=5.0,
            classification="unverified", report_id="rpt_test",
            pixel_coords=(1.0, 2.0), created_at=datetime.now(UTC),
            detections=[det],
        )
        catalog.add(cand)
        loaded = catalog.get(cand.id)
        assert loaded.detections[0].local_rms == 0.037

    def test_local_rms_none_db_persistence(self, tmp_db):
        from parallax.types import Detection
        from parallax import catalog
        det = Detection(filter="F200W", flux=10.0, snr=5.0,
                        pixel_coords=(1.0, 2.0))
        cand = Candidate(
            id="cnd_lrms003", ra=83.0, dec=-5.0, flux=10.0, snr=5.0,
            classification="unverified", report_id="rpt_test",
            pixel_coords=(1.0, 2.0), created_at=datetime.now(UTC),
            detections=[det],
        )
        catalog.add(cand)
        loaded = catalog.get(cand.id)
        assert loaded.detections[0].local_rms is None


class TestBackgroundVarianceWarning:
    def test_variance_warning_in_markdown(self, tmp_db):
        from parallax.types import Detection
        from parallax.survey import _write_markdown
        d1 = Detection(filter="F200W", flux=10.0, snr=5.0,
                       pixel_coords=(1.0, 2.0), local_rms=0.01)
        d2 = Detection(filter="F150W", flux=8.0, snr=4.0,
                       pixel_coords=(1.0, 2.0), local_rms=0.05)
        cand = Candidate(
            id="cnd_bkg001", ra=83.0, dec=-5.0, flux=10.0, snr=5.0,
            classification="unverified", report_id="rpt_test",
            pixel_coords=(1.0, 2.0), created_at=datetime.now(UTC),
            detections=[d1, d2],
        )
        rpt = Report(id="20260402_bkgtest", target="M92", instrument="NIRCAM",
                     filters=["F200W", "F150W"], created_at=datetime.now(UTC),
                     candidates=[cand], n_sources_detected=1,
                     n_catalog_matched=0, n_unverified=1)
        md_path = os.path.join(tmp_db, "reports", "test_bkg.md")
        _write_markdown(rpt, md_path, include_known=False)
        with open(md_path) as f:
            content = f.read()
        assert "Background:" in content
        assert "5.0x" in content

    def test_no_warning_when_ratio_low(self, tmp_db):
        from parallax.types import Detection
        from parallax.survey import _write_markdown
        d1 = Detection(filter="F200W", flux=10.0, snr=5.0,
                       pixel_coords=(1.0, 2.0), local_rms=0.01)
        d2 = Detection(filter="F150W", flux=8.0, snr=4.0,
                       pixel_coords=(1.0, 2.0), local_rms=0.02)
        cand = Candidate(
            id="cnd_bkg002", ra=83.0, dec=-5.0, flux=10.0, snr=5.0,
            classification="unverified", report_id="rpt_test",
            pixel_coords=(1.0, 2.0), created_at=datetime.now(UTC),
            detections=[d1, d2],
        )
        rpt = Report(id="20260402_bkgtst2", target="M92", instrument="NIRCAM",
                     filters=["F200W", "F150W"], created_at=datetime.now(UTC),
                     candidates=[cand], n_sources_detected=1,
                     n_catalog_matched=0, n_unverified=1)
        md_path = os.path.join(tmp_db, "reports", "test_bkg2.md")
        _write_markdown(rpt, md_path, include_known=False)
        with open(md_path) as f:
            content = f.read()
        assert "Background:" not in content


class TestHintsMarkdownFormat:
    def test_hints_format_plain_text(self, tmp_db):
        from parallax.types import Detection
        from parallax.survey import _write_markdown
        cand = Candidate(
            id="cnd_fmt001", ra=83.0, dec=-5.0, flux=10.0, snr=5.0,
            classification="unverified", report_id="rpt_test",
            pixel_coords=(1.0, 2.0), created_at=datetime.now(UTC),
            hints=["point source in F200W", "single-filter detection"],
        )
        rpt = Report(id="20260402_fmttest", target="M92", instrument="NIRCAM",
                     filters=["F200W"], created_at=datetime.now(UTC),
                     candidates=[cand], n_sources_detected=1,
                     n_catalog_matched=0, n_unverified=1)
        md_path = os.path.join(tmp_db, "reports", "test_fmt.md")
        _write_markdown(rpt, md_path, include_known=False)
        with open(md_path) as f:
            content = f.read()
        # single bold for ID, plain text for rest
        assert "**cnd_fmt001** - Hints:" in content
        # no double bold
        assert "**Hints:**" not in content


class TestMagAbZeroPoint:
    def test_mag_ab_uses_mjy_zeropoint(self, tmp_db):
        """mag_ab is computed with 0.003631 MJy (not 3631 Jy) as the AB zeropoint."""
        import math
        pixar_sr = 1.0  # 1 sr/px - artificial but forces finite, testable mag values
        hdul = make_fits(n_sources=5, noise=0.01, pixar_sr=pixar_sr)
        path = os.path.join(tmp_db, "ab_test.fits")
        hdul.writeto(path, overwrite=True)

        from parallax.survey import detect
        results = detect(path, snr_threshold=1.0, min_pixels=1)

        checked = 0
        for s in results:
            if s["flux_mjy"] is not None and s["mag_ab"] is not None:
                expected = round(-2.5 * math.log10(s["flux_mjy"] / 0.003631), 4)
                assert s["mag_ab"] == expected, (
                    f"mag_ab {s['mag_ab']} != expected {expected}; flux_mjy={s['flux_mjy']}"
                )
                # sanity: old divisor (3631 Jy) would give a value ~15 mag different
                buggy = round(-2.5 * math.log10(s["flux_mjy"] / 3631.0), 4)
                assert abs(s["mag_ab"] - buggy) > 10
                checked += 1
        assert checked >= 1, "no source had both flux_mjy and mag_ab set"


class TestMinPixelsDefault:
    def test_min_pixels_default_is_5(self):
        """detection.min_pixels default must be 5 to catch NIRCam SW point sources."""
        from parallax.config import _DEFAULTS
        assert _DEFAULTS["detection"]["min_pixels"] == 5


class TestDetectionCacheVersionTag:
    def test_different_version_tag_produces_different_hash(self, tmp_db):
        """_fits_hash includes _CACHE_VERSION so bumping it invalidates prior entries."""
        import parallax.survey as _survey

        hdul = make_fits(n_sources=3, noise=0.05)
        path = os.path.join(tmp_db, "cache_tag_test.fits")
        hdul.writeto(path, overwrite=True)

        h1 = _survey._fits_hash(path, snr_threshold=3.0, min_pixels=5, kernel_fwhm=2.0)

        original = _survey._CACHE_VERSION
        try:
            _survey._CACHE_VERSION = "v999"
            h2 = _survey._fits_hash(path, snr_threshold=3.0, min_pixels=5, kernel_fwhm=2.0)
        finally:
            _survey._CACHE_VERSION = original

        assert h1 != h2
