import json
import os
from datetime import datetime, UTC

import pytest

from parallax.types import Candidate, CatalogMatch


def _make_candidate(id="cnd_00000001", ra=83.8221, dec=-5.3911, cls="unverified",
                    tags=None, notes=None, matches=None):
    return Candidate(
        id=id, ra=ra, dec=dec, flux=150.0, snr=6.0,
        classification=cls, report_id="rpt_20260101_aabbccdd",
        pixel_coords=(100.0, 100.0), created_at=datetime.now(UTC),
        catalog_matches=matches or [],
        tags=tags or [],
        notes=notes or [],
    )


class TestAddGet:
    def test_roundtrip(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate(
            matches=[CatalogMatch("SIMBAD", "star1", 0.5, "Star", 0.03, {"mag": 12.5})]
        )
        catalog.add(cand)
        got = catalog.get(cand.id)
        assert got is not None
        assert got.id == cand.id
        assert got.ra == cand.ra
        assert got.classification == "unverified"
        assert len(got.catalog_matches) == 1
        assert got.catalog_matches[0].catalog == "SIMBAD"
        assert got.catalog_matches[0].redshift == 0.03

    def test_get_missing(self, tmp_db):
        from parallax import catalog
        assert catalog.get("cnd_nonexist") is None

    def test_duplicate_raises(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate()
        catalog.add(cand)
        with pytest.raises(ValueError):
            catalog.add(cand)


class TestQuery:
    def test_within_radius(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate(ra=83.8221, dec=-5.3911)
        catalog.add(cand)

        results = catalog.query(83.8221, -5.3911, 2.0)
        assert len(results) == 1
        assert results[0].id == cand.id

    def test_outside_radius(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate(ra=83.8221, dec=-5.3911)
        catalog.add(cand)

        # 10 arcsec away is too far for a 2 arcsec search
        results = catalog.query(83.8221, -5.3911 + 10.0/3600, 2.0)
        assert len(results) == 0

    def test_radius_157(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate(id="cnd_rad157", ra=10.0, dec=20.0)
        catalog.add(cand)
        results = catalog.query(10.0, 20.0, 15.7)
        assert len(results) == 1

    def test_classification_filter(self, tmp_db):
        from parallax import catalog
        catalog.add(_make_candidate(id="cnd_q1", cls="unverified"))
        catalog.add(_make_candidate(id="cnd_q2", cls="known"))
        results = catalog.query(83.8221, -5.3911, 5.0, classification="known")
        assert all(c.classification == "known" for c in results)


class TestUpdate:
    def test_classification_change(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate()
        catalog.add(cand)
        updated = catalog.update(cand.id, classification="known")
        assert updated.classification == "known"

    def test_disallowed_field(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate()
        catalog.add(cand)
        with pytest.raises(ValueError):
            catalog.update(cand.id, flux=999.0)

    def test_tags_replace(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate(tags=["old"])
        catalog.add(cand)
        updated = catalog.update(cand.id, tags=["new", "fresh"])
        assert updated.tags == ["new", "fresh"]

    def test_empty_tags(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate(id="cnd_emptytag", tags=["something"])
        catalog.add(cand)
        updated = catalog.update(cand.id, tags=[])
        assert updated.tags == []


class TestHistory:
    def test_records_change(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate()
        catalog.add(cand)
        catalog.update(cand.id, classification="known")
        h = catalog.history(cand.id)
        assert len(h) == 1
        assert h[0]["field"] == "classification"
        assert h[0]["old_value"] == "unverified"
        assert h[0]["new_value"] == "known"

    def test_missing_raises(self, tmp_db):
        from parallax import catalog
        with pytest.raises(KeyError):
            catalog.history("cnd_nope")


class TestDelete:
    def test_delete_then_get(self, tmp_db):
        from parallax import catalog
        cand = _make_candidate()
        catalog.add(cand)
        catalog.delete(cand.id)
        assert catalog.get(cand.id) is None

    def test_delete_missing(self, tmp_db):
        from parallax import catalog
        with pytest.raises(KeyError):
            catalog.delete("cnd_ghost")


class TestList:
    def test_limit_offset(self, tmp_db):
        from parallax import catalog
        for i in range(5):
            catalog.add(_make_candidate(id=f"cnd_list{i:04d}"))

        first_page = catalog.list(limit=2, offset=0)
        assert len(first_page) == 2
        second_page = catalog.list(limit=2, offset=2)
        assert len(second_page) == 2
        ids1 = {c.id for c in first_page}
        ids2 = {c.id for c in second_page}
        assert ids1.isdisjoint(ids2)

    def test_filter_by_tag(self, tmp_db):
        from parallax import catalog
        catalog.add(_make_candidate(id="cnd_tagged", tags=["followup"]))
        catalog.add(_make_candidate(id="cnd_notag"))

        results = catalog.list(tags=["followup"])
        assert len(results) == 1
        assert results[0].id == "cnd_tagged"
