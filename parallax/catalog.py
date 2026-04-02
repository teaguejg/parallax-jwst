import builtins
import json
import math
import logging
from datetime import datetime, UTC

from astropy.coordinates import SkyCoord

from parallax._db import get_db
from parallax.types import Candidate, CatalogMatch, Detection, _now_iso, _safe_json_dict

logger = logging.getLogger(__name__)

_ALLOWED_FIELDS = {"classification", "tags", "notes", "confidence"}
_VALID_CLASSIFICATIONS = {"unverified", "known-but-notable", "known"}


def add(candidate: Candidate) -> str:
    """Persist a candidate to the database."""
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM candidates WHERE id = ?",
                                (candidate.id,)).fetchone()
        if existing:
            raise ValueError(f"candidate {candidate.id} already exists")

        conn.execute(
            "INSERT INTO candidates (id, ra, dec, flux, snr, classification, "
            "report_id, pixel_x, pixel_y, created_at, tags, notes, confidence, "
            "flux_err, flux_mjy_err, mag_ab_err) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (candidate.id, candidate.ra, candidate.dec, candidate.flux,
             candidate.snr, candidate.classification, candidate.report_id,
             candidate.pixel_coords[0], candidate.pixel_coords[1],
             candidate.created_at.isoformat() if isinstance(candidate.created_at, datetime) else candidate.created_at,
             json.dumps(candidate.tags), json.dumps(candidate.notes),
             candidate.confidence,
             candidate.flux_err, candidate.flux_mjy_err, candidate.mag_ab_err)
        )

        for m in candidate.catalog_matches:
            conn.execute(
                "INSERT INTO catalog_matches (candidate_id, catalog, source_id, "
                "separation_arcsec, object_type, redshift, data) "
                "VALUES (?,?,?,?,?,?,?)",
                (candidate.id, m.catalog, m.source_id, m.separation_arcsec,
                 m.object_type, m.redshift, json.dumps(m.data))
            )

        for det in candidate.detections:
            conn.execute(
                "INSERT INTO candidate_detections "
                "(candidate_id, filter, flux, snr, pixel_x, pixel_y, "
                "flux_mjy, mag_ab, flux_err, flux_mjy_err, mag_ab_err) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (candidate.id, det.filter, det.flux, det.snr,
                 det.pixel_coords[0], det.pixel_coords[1],
                 det.flux_mjy, det.mag_ab,
                 det.flux_err, det.flux_mjy_err, det.mag_ab_err)
            )

    return candidate.id


def _row_to_candidate(row, conn) -> Candidate:
    matches_rows = conn.execute(
        "SELECT * FROM catalog_matches WHERE candidate_id = ?",
        (row["id"],)
    ).fetchall()

    matches = []
    for mr in matches_rows:
        matches.append(CatalogMatch(
            catalog=mr["catalog"],
            source_id=mr["source_id"] or "",
            separation_arcsec=mr["separation_arcsec"] or 0.0,
            object_type=mr["object_type"],
            redshift=mr["redshift"],
            data=_safe_json_dict(mr["data"]),
        ))

    det_rows = conn.execute(
        "SELECT * FROM candidate_detections WHERE candidate_id = ?",
        (row["id"],)
    ).fetchall()
    detections = []
    for dr in det_rows:
        dk = dr.keys()
        kw = {}
        if "flux_mjy" in dk:
            kw["flux_mjy"] = dr["flux_mjy"]
        if "mag_ab" in dk:
            kw["mag_ab"] = dr["mag_ab"]
        if "flux_err" in dk:
            kw["flux_err"] = dr["flux_err"]
        if "flux_mjy_err" in dk:
            kw["flux_mjy_err"] = dr["flux_mjy_err"]
        if "mag_ab_err" in dk:
            kw["mag_ab_err"] = dr["mag_ab_err"]
        detections.append(Detection(
            filter=dr["filter"], flux=dr["flux"], snr=dr["snr"],
            pixel_coords=(dr["pixel_x"], dr["pixel_y"]), **kw
        ))

    created = row["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created)

    keys = row.keys()
    return Candidate(
        id=row["id"],
        ra=row["ra"],
        dec=row["dec"],
        flux=row["flux"],
        snr=row["snr"],
        classification=row["classification"],
        report_id=row["report_id"],
        pixel_coords=(row["pixel_x"] or 0.0, row["pixel_y"] or 0.0),
        created_at=created,
        catalog_matches=matches,
        detections=detections,
        tags=json.loads(row["tags"]) if row["tags"] else [],
        notes=json.loads(row["notes"]) if row["notes"] else [],
        confidence=row["confidence"] if "confidence" in keys else 0.0,
        flux_err=row["flux_err"] if "flux_err" in keys else None,
        flux_mjy_err=row["flux_mjy_err"] if "flux_mjy_err" in keys else None,
        mag_ab_err=row["mag_ab_err"] if "mag_ab_err" in keys else None,
    )


def get(candidate_id: str) -> Candidate | None:
    """Retrieve a candidate by ID, or None if not found."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?",
                           (candidate_id,)).fetchone()
        if row is None:
            return None
        return _row_to_candidate(row, conn)


def query(
    ra: float,
    dec: float,
    radius_arcsec: float,
    classification: str | None = None,
) -> builtins.list[Candidate]:
    """Find candidates within a sky radius."""
    # TODO: use proper spherical geometry for large radii
    r = radius_arcsec / 3600.0
    with get_db() as conn:
        sql = ("SELECT * FROM candidates WHERE ra = ra AND "  # filters NaN
               "ra BETWEEN ? AND ? AND dec BETWEEN ? AND ?")
        params: builtins.list = [ra - r, ra + r, dec - r, dec + r]

        if classification:
            sql += " AND classification = ?"
            params.append(classification)

        rows = conn.execute(sql, params).fetchall()

        center = SkyCoord(ra, dec, unit="deg")
        keep = []
        for row in rows:
            if row["ra"] is None or math.isnan(row["ra"]):
                continue
            pos = SkyCoord(row["ra"], row["dec"], unit="deg")
            if center.separation(pos).arcsec <= radius_arcsec:
                keep.append(_row_to_candidate(row, conn))

    return keep


def update(candidate_id: str, **kwargs) -> Candidate:
    """Update mutable fields on a candidate."""
    for k in kwargs:
        if k not in _ALLOWED_FIELDS:
            raise ValueError(f"cannot update field: {k}")

    if "classification" in kwargs:
        if kwargs["classification"] not in _VALID_CLASSIFICATIONS:
            raise ValueError(f"invalid classification: {kwargs['classification']}")

    with get_db() as conn:
        row = conn.execute("SELECT * FROM candidates WHERE id = ?",
                           (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)

        now = _now_iso()

        for field, new_val in kwargs.items():
            if field in ("tags", "notes"):
                old_val = json.loads(row[field]) if row[field] else []
                conn.execute(
                    "INSERT INTO candidate_history (candidate_id, timestamp, field, old_value, new_value) "
                    "VALUES (?,?,?,?,?)",
                    (candidate_id, now, field, json.dumps(old_val), json.dumps(new_val))
                )
                conn.execute(f"UPDATE candidates SET {field} = ? WHERE id = ?",
                             (json.dumps(new_val), candidate_id))
            else:
                old_val = row[field]
                conn.execute(
                    "INSERT INTO candidate_history (candidate_id, timestamp, field, old_value, new_value) "
                    "VALUES (?,?,?,?,?)",
                    (candidate_id, now, field, json.dumps(old_val), json.dumps(new_val))
                )
                conn.execute(f"UPDATE candidates SET {field} = ? WHERE id = ?",
                             (new_val, candidate_id))

    return get(candidate_id)


def history(candidate_id: str) -> builtins.list[dict]:
    """Return the change history for a candidate."""
    with get_db() as conn:
        row = conn.execute("SELECT id FROM candidates WHERE id = ?",
                           (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)

        rows = conn.execute(
            "SELECT * FROM candidate_history WHERE candidate_id = ? ORDER BY timestamp ASC",
            (candidate_id,)
        ).fetchall()

    result = []
    for r in rows:
        result.append({
            "timestamp": r["timestamp"],
            "field": r["field"],
            "old_value": json.loads(r["old_value"]) if r["old_value"] else None,
            "new_value": json.loads(r["new_value"]) if r["new_value"] else None,
        })
    return result


def list(
    classification: str | None = None,
    tags: builtins.list[str] | None = None,
    limit: int = 100,
    offset: int = 0,
) -> builtins.list[Candidate]:
    """List candidates with optional filters, newest first."""
    with get_db() as conn:
        sql = "SELECT * FROM candidates WHERE 1=1"
        params: builtins.list = []

        if classification:
            sql += " AND classification = ?"
            params.append(classification)

        if tags:
            for t in tags:
                sql += " AND tags LIKE ?"
                params.append(f'%"{t}"%')

        sql += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(sql, params).fetchall()
        return [_row_to_candidate(r, conn) for r in rows]


def delete(candidate_id: str) -> None:
    """Hard-delete a candidate, its catalog matches, and its history."""
    with get_db() as conn:
        row = conn.execute("SELECT id FROM candidates WHERE id = ?",
                           (candidate_id,)).fetchone()
        if row is None:
            raise KeyError(candidate_id)

        conn.execute("DELETE FROM candidate_history WHERE candidate_id = ?",
                     (candidate_id,))
        conn.execute("DELETE FROM catalog_matches WHERE candidate_id = ?",
                     (candidate_id,))
        conn.execute("DELETE FROM candidates WHERE id = ?",
                     (candidate_id,))
