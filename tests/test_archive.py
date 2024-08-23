import csv
import json
import os
import tempfile
from datetime import datetime, timedelta, UTC

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from parallax.types import Candidate, CatalogMatch, Report, report_to_dict
from parallax import catalog, archive
from conftest import make_fits


def _make_candidate(report_id="rpt_20260101_aabbccdd", suffix="01", classification="unverified"):
    return Candidate(
        id=f"cnd_test{suffix}",
        ra=83.8221, dec=-5.3911,
        flux=1500.0, snr=5.5,
        classification=classification,
        report_id=report_id,
        pixel_coords=(100.0, 100.0),
        created_at=datetime(2026, 1, 15, 12, 0),
        catalog_matches=[
            CatalogMatch(catalog="SIMBAD", source_id="HD 12345",
                         separation_arcsec=0.5, object_type="Star"),
        ],
        tags=["bright"],
        notes=["interesting source"],
    )


def _make_report(tmp_dir, report_id="rpt_20260101_aabbccdd", target="Orion"):
    hdul = make_fits()
    fits_path = os.path.join(tmp_dir, "downloads", "test.fits")
    hdul.writeto(fits_path, overwrite=True)
    hdul.close()

    cand = _make_candidate(report_id)
    rpt = Report(
        id=report_id, target=target,
        instrument="NIRCAM", filters=["F200W"],
        created_at=datetime(2026, 1, 15, 12, 0),
        candidates=[cand],
        n_sources_detected=5, n_catalog_matched=1, n_unverified=1,
    )
    rpt._fits_path = fits_path  # stash for test helpers
    return rpt, cand


def _persist_report(rpt, conn):
    fp = getattr(rpt, "_fits_path", None)
    conn.execute(
        "INSERT INTO reports (id, target, instrument, filter, observation_id, "
        "fits_path, created_at, n_sources_detected, n_catalog_matched, n_unverified, "
        "json_path, md_path, filters) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (rpt.id, rpt.target, rpt.instrument,
         rpt.filters[0] if rpt.filters else None,
         "TEST001", fp,
         rpt.created_at.isoformat(), rpt.n_sources_detected,
         rpt.n_catalog_matched, rpt.n_unverified, rpt.json_path, rpt.md_path,
         json.dumps(rpt.filters))
    )


class TestSearch:
    def test_find_report_by_target(self, tmp_db):
        from parallax._db import get_db
        rpt, cand = _make_report(tmp_db)
        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)

        result = archive.search("orion")
        assert len(result["reports"]) >= 1
        assert result["reports"][0].target == "Orion"

    def test_find_candidate_by_classification(self, tmp_db):
        rpt, cand = _make_report(tmp_db)
        from parallax._db import get_db
        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)

        result = archive.search_candidates("unverified", field="classification")
        assert any(c.id == cand.id for c in result)

    def test_search_empty(self, tmp_db):
        result = archive.search("zzz_nonexistent_zzz")
        assert result["reports"] == []
        assert result["candidates"] == []


class TestReports:
    def test_reports_newest_first(self, tmp_db):
        from parallax._db import get_db

        rpt1, c1 = _make_report(tmp_db, "rpt_20260101_aaaa0001", "Alpha")
        rpt2_id = "rpt_20260115_aaaa0002"
        c2 = _make_candidate(rpt2_id, suffix="02")
        rpt2 = Report(
            id=rpt2_id, target="Beta",
            instrument="NIRCAM", filters=["F200W"],
            created_at=datetime(2026, 1, 20, 12, 0),
            candidates=[c2],
            n_sources_detected=3, n_catalog_matched=0, n_unverified=1,
        )
        rpt2._fits_path = rpt1._fits_path

        with get_db() as conn:
            _persist_report(rpt1, conn)
            _persist_report(rpt2, conn)
        catalog.add(c1)
        catalog.add(c2)

        listed = archive.reports()
        assert len(listed) == 2
        assert listed[0].id == rpt2_id  # newer

    def test_get_report_by_id(self, tmp_db):
        from parallax._db import get_db
        rpt, cand = _make_report(tmp_db)
        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)

        loaded = archive.get_report(rpt.id)
        assert loaded is not None
        assert loaded.target == "Orion"

    def test_get_report_none(self, tmp_db):
        assert archive.get_report("rpt_fake_00000000") is None

    def test_get_report_prefers_json(self, tmp_db):
        from parallax._db import get_db
        rpt, cand = _make_report(tmp_db)

        json_path = os.path.join(tmp_db, "reports", f"{rpt.id}.json")
        rpt.json_path = json_path
        d = report_to_dict(rpt)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(d, f)

        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)

        loaded = archive.get_report(rpt.id)
        assert loaded is not None
        assert loaded.target == "Orion"


class TestGetFits:
    def test_cached_cutout(self, tmp_db):
        cand = _make_candidate()
        catalog.add(cand)

        cache_dir = os.path.join(tmp_db, "archive", "cutouts")
        cache_path = os.path.join(cache_dir, f"{cand.id}.fits")
        os.makedirs(cache_dir, exist_ok=True)

        # write a tiny fits as the cached cutout
        hdr = fits.Header()
        fits.writeto(cache_path, np.zeros((10, 10)), hdr, overwrite=True)

        result = archive.get_fits(cand.id)
        assert result == cache_path
        assert os.path.isfile(result)

    def test_extracts_from_original(self, tmp_db):
        from parallax._db import get_db
        rpt, cand = _make_report(tmp_db)
        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)

        path = archive.get_fits(cand.id)
        assert path.endswith(".fits")
        assert os.path.isfile(path)

    def test_missing_candidate_raises(self, tmp_db):
        with pytest.raises(KeyError):
            archive.get_fits("cnd_nope0000")


class TestExport:
    def test_csv_headers(self, tmp_db):
        from parallax._db import get_db
        rpt, cand = _make_report(tmp_db)
        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)

        out = archive.export(rpt.id, format="csv")
        assert out.endswith(".csv")
        with open(out) as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert "id" in headers
        assert "ra" in headers
        assert "catalog_matches" in headers

    def test_json_valid(self, tmp_db):
        from parallax._db import get_db
        rpt, cand = _make_report(tmp_db)
        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)

        out = archive.export(rpt.id, format="json")
        with open(out) as f:
            data = json.load(f)
        assert data["id"] == rpt.id
        assert len(data["candidates"]) == 1

    def test_unknown_format(self, tmp_db):
        from parallax._db import get_db
        rpt, cand = _make_report(tmp_db)
        with get_db() as conn:
            _persist_report(rpt, conn)
        catalog.add(cand)
        with pytest.raises(ValueError):
            archive.export(rpt.id, format="xml")

    def test_missing_report_raises(self, tmp_db):
        with pytest.raises(KeyError):
            archive.export("rpt_fake_00000000")


class TestTag:
    def test_merge_no_duplicates(self, tmp_db):
        cand = _make_candidate()
        catalog.add(cand)

        updated = archive.tag(cand.id, ["bright", "followup"])
        assert "bright" in updated.tags
        assert "followup" in updated.tags
        assert updated.tags.count("bright") == 1

    def test_tag_string_input(self, tmp_db):
        cand = _make_candidate()
        catalog.add(cand)
        updated = archive.tag(cand.id, "new_tag")
        assert "new_tag" in updated.tags

    def test_tag_not_found(self, tmp_db):
        with pytest.raises(KeyError):
            archive.tag("cnd_nope0000", "x")


class TestAnnotate:
    def test_appends_note(self, tmp_db):
        cand = _make_candidate()
        catalog.add(cand)

        updated = archive.annotate(cand.id, "second note")
        assert "interesting source" in updated.notes
        assert "second note" in updated.notes
        assert len(updated.notes) == 2

    def test_annotate_not_found(self, tmp_db):
        with pytest.raises(KeyError):
            archive.annotate("cnd_nope0000", "note")


class TestPrune:
    def test_dry_run_no_delete(self, tmp_db):
        # create an old file
        dl_dir = os.path.join(tmp_db, "downloads")
        old_file = os.path.join(dl_dir, "old.fits")
        with open(old_file, "wb") as f:
            f.write(b"x" * 1024)
        # set mtime to 120 days ago
        old_ts = (datetime.now(UTC) - timedelta(days=120)).timestamp()
        os.utime(old_file, (old_ts, old_ts))

        result = archive.prune(older_than_days=90, dry_run=True)
        assert result["files_removed"] >= 1
        assert result["bytes_freed"] >= 1024
        assert os.path.isfile(old_file)  # not actually deleted

    def test_prune_actually_deletes(self, tmp_db):
        dl_dir = os.path.join(tmp_db, "downloads")
        old_file = os.path.join(dl_dir, "ancient.fits")
        with open(old_file, "wb") as f:
            f.write(b"y" * 512)
        old_ts = (datetime.now(UTC) - timedelta(days=200)).timestamp()
        os.utime(old_file, (old_ts, old_ts))

        result = archive.prune(older_than_days=90, dry_run=False)
        assert not os.path.isfile(old_file)


# oddly specific test -- came from a bug where tag with empty existing tags crashed
def test_tag_on_candidate_with_no_tags(tmp_db):
    cand = Candidate(
        id="cnd_notags1", ra=10.0, dec=20.0,
        flux=100.0, snr=3.0, classification="unverified",
        report_id="rpt_20260101_xxxxxxxx",
        pixel_coords=(50.0, 50.0),
        created_at=datetime(2026, 3, 1),
        tags=[], notes=[],
    )
    catalog.add(cand)
    updated = archive.tag(cand.id, "first")
    assert updated.tags == ["first"]


def test_annotate_dedup(tmp_db):
    cand = Candidate(
        id="cnd_notedup", ra=10.0, dec=20.0,
        flux=100.0, snr=3.0, classification="unverified",
        report_id="rpt_20260101_xxxxxxxx",
        pixel_coords=(50.0, 50.0),
        created_at=datetime(2026, 3, 1),
        tags=[], notes=[],
    )
    catalog.add(cand)
    archive.annotate(cand.id, "first note")
    archive.annotate(cand.id, "first note")  # same note again
    archive.annotate(cand.id, "second note")

    result = catalog.get(cand.id)
    assert result.notes == ["first note", "second note"]
