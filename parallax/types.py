from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, UTC
import re
import uuid
import numpy as np
from astropy.wcs import WCS


@dataclass
class CatalogMatch:
    catalog: str                    # "SIMBAD" | "NED" | "GAIA"
    source_id: str
    separation_arcsec: float
    object_type: str | None = None
    redshift: float | None = None
    data: dict = field(default_factory=dict)


@dataclass
class Detection:
    filter: str
    flux: float
    snr: float
    pixel_coords: tuple[float, float]
    flux_mjy: float | None = None
    mag_ab: float | None = None


@dataclass
class Candidate:
    id: str                         # format: "cnd_{8-char hex}"
    ra: float                       # degrees; NaN if WCS failed
    dec: float                      # degrees; NaN if WCS failed
    flux: float                     # kron_flux in instrumental units
    snr: float
    classification: str             # "unverified" | "known"
    report_id: str
    pixel_coords: tuple[float, float]   # (x, y) centroid in source FITS
    created_at: datetime
    catalog_matches: list[CatalogMatch] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class Report:
    id: str                         # format: "{YYYYMMDD}_{8-char fingerprint or hex}"
    target: str
    instrument: str                 # from FITS header INSTRUME
    filters: list[str] = field(default_factory=list)
    created_at: datetime = field(default=None)
    candidates: list[Candidate] = field(default_factory=list)
    n_sources_detected: int = 0
    n_catalog_matched: int = 0
    n_unverified: int = 0
    json_path: str | None = None
    md_path: str | None = None


@dataclass
class Criteria:
    name: str
    instruments: list[str]
    filters: list[str] | None = None
    ra: float | None = None
    dec: float | None = None
    radius_deg: float | None = None
    check_interval_minutes: int = 60


@dataclass
class CutoutView:
    candidate: Candidate
    data: np.ndarray                # 2D float array, background-subtracted
    wcs: WCS
    fits_path: str
    shape: tuple[int, int]


class ViewSession:
    """Holds loaded FITS data and report for repeated examine() calls."""
    def __init__(self, report: Report, fits):
        self.report = report
        self.fits = fits            # astropy.io.fits.HDUList
        self.candidates = report.candidates


def _candidate_id() -> str:
    return f"cnd_{uuid.uuid4().hex[:8]}"


def _report_id(target: str, fingerprint: str | None = None) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%d")
    token = fingerprint[:8] if fingerprint else uuid.uuid4().hex[:8]
    return f"{ts}_{token}"


def _target_slug(target: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', target.lower()).strip('_')


def _watch_id() -> str:
    return f"watch_{uuid.uuid4().hex[:8]}"


def _now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(UTC).isoformat()


def report_to_dict(r: Report) -> dict:
    from dataclasses import asdict
    d = asdict(r)
    d["created_at"] = r.created_at.isoformat()
    d["filters"] = list(r.filters)
    for c in d["candidates"]:
        if isinstance(c["created_at"], datetime):
            c["created_at"] = c["created_at"].isoformat()
        c["detections"] = [
            {"filter": det["filter"], "flux": det["flux"],
             "snr": det["snr"], "pixel_coords": det["pixel_coords"],
             "flux_mjy": det.get("flux_mjy"), "mag_ab": det.get("mag_ab")}
            for det in c.get("detections", [])
        ]
    return d


def report_from_dict(d: dict) -> Report:
    candidates = []
    for cd in d.get("candidates", []):
        matches = []
        for m in cd.get("catalog_matches", []):
            matches.append(CatalogMatch(
                catalog=m["catalog"],
                source_id=m["source_id"],
                separation_arcsec=m["separation_arcsec"],
                object_type=m.get("object_type"),
                redshift=m.get("redshift"),
                data=m.get("data", {}),
            ))

        dets = []
        for det in cd.get("detections", []):
            px = det.get("pixel_coords", (0.0, 0.0))
            if isinstance(px, list):
                px = tuple(px)
            dets.append(Detection(
                filter=det["filter"], flux=det["flux"],
                snr=det["snr"], pixel_coords=px,
                flux_mjy=det.get("flux_mjy"),
                mag_ab=det.get("mag_ab"),
            ))

        created = cd["created_at"]
        if isinstance(created, str):
            created = datetime.fromisoformat(created)

        px = cd.get("pixel_coords", (0.0, 0.0))
        if isinstance(px, list):
            px = tuple(px)

        candidates.append(Candidate(
            id=cd["id"],
            ra=cd["ra"],
            dec=cd["dec"],
            flux=cd["flux"],
            snr=cd["snr"],
            classification=("known" if cd["classification"] == "known-but-notable"
                            else cd["classification"]),
            report_id=cd["report_id"],
            pixel_coords=px,
            created_at=created,
            catalog_matches=matches,
            detections=dets,
            tags=cd.get("tags", []),
            notes=cd.get("notes", []),
            confidence=cd.get("confidence", 0.0),
        ))

    created = d["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created)

    return Report(
        id=d["id"],
        target=d["target"],
        instrument=d["instrument"],
        filters=d.get("filters", [d["filter"]] if "filter" in d else []),
        created_at=created,
        candidates=candidates,
        n_sources_detected=d["n_sources_detected"],
        n_catalog_matched=d["n_catalog_matched"],
        n_unverified=d["n_unverified"],
        json_path=d.get("json_path"),
        md_path=d.get("md_path"),
    )
