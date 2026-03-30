import csv
import json
import logging
import os
from datetime import datetime, UTC

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u

from parallax.config import config
from parallax._db import get_db
from parallax.types import (
    Candidate, CatalogMatch, Detection, Report,
    report_to_dict, report_from_dict,
)
from parallax import catalog

logger = logging.getLogger(__name__)


def _normalize_target(name: str) -> str:
    if not name:
        return name
    # fix common catalog prefixes that get title-cased incorrectly
    prefixes = ["Ngc", "Ic", "Hd", "Hr", "Bd", "Pgc", "Ugc", "Mcg"]
    parts = name.split()
    if parts and parts[0] in prefixes:
        parts[0] = parts[0].upper()
        return " ".join(parts)
    return name


def _row_to_report(row, conn) -> Report:
    cand_rows = conn.execute(
        "SELECT * FROM candidates WHERE report_id = ? ORDER BY created_at DESC",
        (row["id"],)
    ).fetchall()

    candidates = []
    for cr in cand_rows:
        matches_rows = conn.execute(
            "SELECT * FROM catalog_matches WHERE candidate_id = ?",
            (cr["id"],)
        ).fetchall()
        matches = []
        for mr in matches_rows:
            matches.append(CatalogMatch(
                catalog=mr["catalog"],
                source_id=mr["source_id"] or "",
                separation_arcsec=mr["separation_arcsec"] or 0.0,
                object_type=mr["object_type"],
                redshift=mr["redshift"],
                data=json.loads(mr["data"]) if mr["data"] else {},
            ))

        det_rows = conn.execute(
            "SELECT * FROM candidate_detections WHERE candidate_id = ?",
            (cr["id"],)
        ).fetchall()
        detections = [
            Detection(filter=dr["filter"], flux=dr["flux"], snr=dr["snr"],
                      pixel_coords=(dr["pixel_x"], dr["pixel_y"]))
            for dr in det_rows
        ]

        created = cr["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)

        candidates.append(Candidate(
            id=cr["id"],
            ra=cr["ra"], dec=cr["dec"],
            flux=cr["flux"], snr=cr["snr"],
            classification=cr["classification"],
            report_id=cr["report_id"],
            pixel_coords=(cr["pixel_x"] or 0.0, cr["pixel_y"] or 0.0),
            created_at=created,
            catalog_matches=matches,
            detections=detections,
            tags=json.loads(cr["tags"]) if cr["tags"] else [],
            notes=json.loads(cr["notes"]) if cr["notes"] else [],
            confidence=cr["confidence"] if cr["confidence"] is not None else 0.0,
        ))

    rpt_created = row["created_at"]
    if isinstance(rpt_created, str):
        rpt_created = datetime.fromisoformat(rpt_created)

    # resolve filters: prefer report_inputs table, fall back to DB columns
    input_rows = conn.execute(
        "SELECT filter FROM report_inputs WHERE report_id = ? ORDER BY id",
        (row["id"],)
    ).fetchall()
    if input_rows:
        filter_list = [ir["filter"] for ir in input_rows]
    else:
        # old report -- try filters JSON column, then single filter column
        raw_filters = row["filters"] if "filters" in row.keys() else None
        if raw_filters:
            try:
                filter_list = json.loads(raw_filters)
            except (json.JSONDecodeError, TypeError):
                filter_list = [row["filter"]] if row["filter"] else []
        else:
            filter_list = [row["filter"]] if row["filter"] else []

    return Report(
        id=row["id"],
        target=_normalize_target(row["target"]),
        instrument=row["instrument"],
        filters=filter_list,
        created_at=rpt_created,
        candidates=candidates,
        n_sources_detected=row["n_sources_detected"],
        n_catalog_matched=row["n_catalog_matched"],
        n_unverified=row["n_unverified"],
        json_path=row["json_path"],
        md_path=row["md_path"],
    )


def search(query: str, field: str | None = None) -> dict:
    """Search reports and candidates by text."""
    return {
        "reports": search_reports(query, field),
        "candidates": search_candidates(query, field),
    }


def search_reports(query: str, field: str | None = None) -> list[Report]:
    """Search only reports."""
    q = query.lower()
    with get_db() as conn:
        if field == "target" or field is None:
            rows = conn.execute(
                "SELECT * FROM reports WHERE LOWER(target) LIKE ?",
                (f"%{q}%",)
            ).fetchall()
        else:
            rows = []

        found = [_row_to_report(r, conn) for r in rows]

    if field is None:
        seen_ids = {r.id for r in found}
        with get_db() as conn:
            all_rows = conn.execute("SELECT * FROM reports").fetchall()
            for row in all_rows:
                if row["id"] in seen_ids:
                    continue
                mp = row["md_path"]
                if mp and os.path.isfile(mp):
                    try:
                        with open(mp, "r") as f:
                            if q in f.read().lower():
                                found.append(_row_to_report(row, conn))
                    except Exception:
                        pass

    return found


def search_candidates(query: str, field: str | None = None) -> list[Candidate]:
    """Search only candidates."""
    q = query.lower()
    clauses = []
    if field == "tags" or field is None:
        clauses.append("LOWER(tags) LIKE ?")
    if field == "notes" or field is None:
        clauses.append("LOWER(notes) LIKE ?")
    if field == "classification" or field is None:
        clauses.append("LOWER(classification) LIKE ?")

    if not clauses:
        return []

    where = " OR ".join(clauses)
    params = [f"%{q}%" for _ in clauses]

    with get_db() as conn:
        rows = conn.execute(
            f"SELECT * FROM candidates WHERE {where}", params
        ).fetchall()

        results = []
        for row in rows:
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
                    data=json.loads(mr["data"]) if mr["data"] else {},
                ))
            created = row["created_at"]
            if isinstance(created, str):
                created = datetime.fromisoformat(created)
            results.append(Candidate(
                id=row["id"], ra=row["ra"], dec=row["dec"],
                flux=row["flux"], snr=row["snr"],
                classification=row["classification"],
                report_id=row["report_id"],
                pixel_coords=(row["pixel_x"] or 0.0, row["pixel_y"] or 0.0),
                created_at=created, catalog_matches=matches,
                tags=json.loads(row["tags"]) if row["tags"] else [],
                notes=json.loads(row["notes"]) if row["notes"] else [],
            ))

    return results


def tag(candidate_id: str, tags: list[str] | str) -> Candidate:
    """Add tags to a candidate (additive, no duplicates)."""
    if isinstance(tags, str):
        tags = [tags]

    existing = catalog.get(candidate_id)
    if existing is None:
        raise KeyError(candidate_id)

    merged = list(existing.tags)
    for t in tags:
        if t not in merged:
            merged.append(t)

    return catalog.update(candidate_id, tags=merged)


def annotate(candidate_id: str, note: str) -> Candidate:
    """Append a note to a candidate."""
    existing = catalog.get(candidate_id)
    if existing is None:
        raise KeyError(candidate_id)

    if note in existing.notes:
        return existing
    updated = list(existing.notes) + [note]
    return catalog.update(candidate_id, notes=updated)


def bookmark(candidate_id: str) -> Candidate:
    """Add the bookmarked tag to a candidate."""
    return tag(candidate_id, "bookmarked")


def unbookmark(candidate_id: str) -> Candidate:
    """Remove the bookmarked tag from a candidate."""
    existing = catalog.get(candidate_id)
    if existing is None:
        raise KeyError(candidate_id)
    tags = [t for t in existing.tags if t != "bookmarked"]
    return catalog.update(candidate_id, tags=tags)


def reports(limit: int = 50, offset: int = 0) -> list[Report]:
    """List stored reports, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        ).fetchall()
        return [_row_to_report(r, conn) for r in rows]


def get_report(report_id: str) -> Report | None:
    """Retrieve a single report by ID, or None if not found."""
    with get_db() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?",
                           (report_id,)).fetchone()
        if row is None:
            return None
        return _row_to_report(row, conn)


def get_fits(candidate_id: str) -> str:
    """Return local path to FITS cutout for a candidate."""
    archive_path = config.get("data.archive_path")
    cache_path = os.path.join(archive_path, "cutouts", f"{candidate_id}.fits")

    if os.path.isfile(cache_path):
        return cache_path

    cand = catalog.get(candidate_id)
    if cand is None:
        raise KeyError(candidate_id)

    with get_db() as conn:
        inputs = conn.execute(
            "SELECT fits_path, filter FROM report_inputs WHERE report_id = ? ORDER BY id",
            (cand.report_id,)
        ).fetchall()

        if inputs and cand.detections:
            best_det = max(cand.detections, key=lambda d: d.snr)
            fits_path = None
            for inp in inputs:
                if inp["filter"] == best_det.filter:
                    fits_path = inp["fits_path"]
                    break
            if fits_path is None:
                fits_path = inputs[0]["fits_path"]
        elif inputs:
            fits_path = inputs[0]["fits_path"]
        else:
            # old report fallback
            rpt_row = conn.execute(
                "SELECT fits_path FROM reports WHERE id = ?",
                (cand.report_id,)
            ).fetchone()
            if rpt_row is None or not rpt_row["fits_path"]:
                raise FileNotFoundError(f"no FITS path for report {cand.report_id}")
            fits_path = rpt_row["fits_path"]
    if not os.path.isfile(fits_path):
        # TODO: re-download from MAST if original FITS is gone
        raise FileNotFoundError(f"FITS file missing: {fits_path}")

    size = config.get("detection.cutout_size")
    import math
    if math.isnan(cand.ra) or math.isnan(cand.dec):
        raise FileNotFoundError("candidate has no valid WCS coordinates")

    with fits.open(fits_path) as hdul:
        sci = next((h for h in hdul if h.data is not None and h.data.ndim == 2), None)
        if sci is None:
            raise FileNotFoundError("no 2D science HDU in FITS")
        wcs = WCS(sci.header)
        coord = SkyCoord(cand.ra, cand.dec, unit="deg")
        cutout = Cutout2D(sci.data, coord, size * u.pixel, wcs=wcs)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    fits.writeto(cache_path, cutout.data, cutout.wcs.to_header(), overwrite=True)
    return cache_path


def get_fits_per_filter(candidate_id: str) -> dict[str, str]:
    """Return {filter_name: fits_path} for each filter whose footprint contains the candidate."""
    import warnings
    from astropy.wcs import FITSFixedWarning

    cand = catalog.get(candidate_id)
    if cand is None:
        raise KeyError(candidate_id)

    with get_db() as conn:
        rows = conn.execute(
            "SELECT fits_path, filter FROM report_inputs WHERE report_id = ? ORDER BY id",
            (cand.report_id,)
        ).fetchall()

    # group paths by filter, preserving order
    by_filter = {}
    for row in rows:
        filt, path = row["filter"], row["fits_path"]
        if filt and path and os.path.isfile(path):
            by_filter.setdefault(filt, []).append(path)

    import math
    has_coords = not (math.isnan(cand.ra) or math.isnan(cand.dec))
    coord = SkyCoord(cand.ra, cand.dec, unit="deg") if has_coords else None

    result = {}
    for filt, paths in by_filter.items():
        if coord is None or len(paths) == 1:
            result[filt] = paths[0]
            continue

        matched = None
        for path in paths:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    with fits.open(path) as hdul:
                        sci = next(
                            (h for h in hdul if h.data is not None and h.data.ndim == 2),
                            None,
                        )
                        if sci is None:
                            continue
                        wcs = WCS(sci.header)
                        if wcs.footprint_contains(coord):
                            matched = path
                            break
            except Exception:
                continue

        if matched:
            result[filt] = matched
        else:
            logger.debug("no WCS match for %s filter %s, using first file", candidate_id, filt)
            result[filt] = paths[0]

    return result


def get_fits_for_report(report_id: str) -> dict[str, str]:
    """Return {filter: fits_path} for all inputs of a report that exist on disk."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT filter, fits_path FROM report_inputs WHERE report_id = ? ORDER BY id",
            (report_id,)
        ).fetchall()

    result = {}
    for row in rows:
        filt, path = row["filter"], row["fits_path"]
        if filt and path and filt not in result and os.path.isfile(path):
            result[filt] = path
    return result


def export(report_id: str, format: str = "csv", output_path: str | None = None) -> str:
    """Export a report's candidates to CSV or JSON."""
    rpt = get_report(report_id)
    if rpt is None:
        raise KeyError(report_id)

    if format not in ("csv", "json"):
        raise ValueError(f"unknown format: {format}")

    if output_path is None:
        archive_path = config.get("data.archive_path")
        output_path = os.path.join(archive_path, f"{report_id}.{format}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    try:
        if format == "csv":
            _write_csv(rpt, output_path)
        else:
            with open(output_path, "w") as f:
                json.dump(report_to_dict(rpt), f, indent=2)
    except OSError as e:
        raise IOError(f"write failed: {e}")

    return output_path


def _write_csv(rpt, path):
    cols = ["id", "ra", "dec", "flux", "snr", "classification",
            "report_id", "pixel_x", "pixel_y", "created_at", "tags", "catalog_matches"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for c in rpt.candidates:
            matches_data = [
                {"catalog": m.catalog, "source_id": m.source_id,
                 "separation_arcsec": m.separation_arcsec,
                 "object_type": m.object_type, "redshift": m.redshift}
                for m in c.catalog_matches
            ]
            w.writerow([
                c.id, c.ra, c.dec, c.flux, c.snr,
                c.classification, c.report_id,
                c.pixel_coords[0], c.pixel_coords[1],
                c.created_at.isoformat() if isinstance(c.created_at, datetime) else c.created_at,
                json.dumps(c.tags),
                json.dumps(matches_data),
            ])


def prune(older_than_days: int = 90, dry_run: bool = True) -> dict:
    """Remove old FITS files and reports from local storage."""
    from datetime import timedelta
    cutoff = datetime.now(UTC) - timedelta(days=older_than_days)
    cutoff_ts = cutoff.isoformat()

    files_removed = 0
    bytes_freed = 0
    reports_removed = 0
    candidates_affected = 0

    for dirkey in ["download_path", "processed_path"]:
        dirpath = config.get(f"data.{dirkey}")
        if not dirpath or not os.path.isdir(dirpath):
            continue
        for root, dirs, files in os.walk(dirpath):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                    if datetime.fromtimestamp(mtime, tz=UTC) < cutoff:
                        sz = os.path.getsize(fpath)
                        if not dry_run:
                            os.remove(fpath)
                            logger.info("removed %s (%d bytes)", fpath, sz)
                        files_removed += 1
                        bytes_freed += sz
                except OSError:
                    pass

    with get_db() as conn:
        old_reports = conn.execute(
            "SELECT * FROM reports WHERE created_at < ?", (cutoff_ts,)
        ).fetchall()

        for rr in old_reports:
            n_cands = conn.execute(
                "SELECT COUNT(*) as cnt FROM candidates WHERE report_id = ?",
                (rr["id"],)
            ).fetchone()["cnt"]
            candidates_affected += n_cands
            reports_removed += 1

            if not dry_run:
                conn.execute("DELETE FROM reports WHERE id = ?", (rr["id"],))
                logger.info("removed report %s", rr["id"])

    return {
        "files_removed": files_removed,
        "bytes_freed": bytes_freed,
        "reports_removed": reports_removed,
        "candidates_affected": candidates_affected,
    }
