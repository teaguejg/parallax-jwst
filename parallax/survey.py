import hashlib
import json
import logging
import os
import sys
import math
import warnings
from datetime import datetime, timedelta, UTC

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
import astropy.units as u

warnings.filterwarnings("ignore", category=FITSFixedWarning)

from parallax.config import config
from parallax.exceptions import TargetNotFoundError
from parallax.types import (
    Candidate, CatalogMatch, Detection, Report,
    _candidate_id, _report_id, _target_slug, report_to_dict,
)
from parallax.acquisition import acquire

logger = logging.getLogger(__name__)


def _find_science_hdu(hdul):
    # prefer named SCI extension (JWST i2d format)
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


def _fits_hash(fits_path, snr_threshold, min_pixels, kernel_fwhm):
    mtime = os.path.getmtime(fits_path)
    box_size = config.get("detection.background_box_size", 50)
    box_size_2 = config.get("detection.background_box_size_2", 0)
    filter_size = config.get("detection.background_filter_size", 3)
    interp_mode = config.get("detection.background_interp", "zoom")
    key = f"{fits_path}:{mtime}:{snr_threshold}:{min_pixels}:{kernel_fwhm}:{box_size}:{box_size_2}:{filter_size}:{interp_mode}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _get_detection_cache(h, snr, npix, fwhm):
    from parallax._db import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT detections FROM detection_cache "
            "WHERE fits_hash = ? AND snr_threshold = ? AND min_pixels = ? AND kernel_fwhm = ?",
            (h, snr, npix, fwhm)
        ).fetchone()
    if row:
        return json.loads(row["detections"])
    return None


def _set_detection_cache(fits_path, h, snr, npix, fwhm, detections):
    from parallax._db import get_db
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO detection_cache "
            "(fits_path, fits_hash, snr_threshold, min_pixels, kernel_fwhm, detections, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (fits_path, h, snr, npix, fwhm, json.dumps(detections), datetime.now(UTC).isoformat())
        )


# TODO: add LRU eviction if cache table grows large
def _get_catalog_cache(field_key):
    from parallax._db import get_db
    now = datetime.now(UTC).isoformat()
    with get_db() as conn:
        row = conn.execute(
            "SELECT results, expires_at FROM catalog_cache WHERE field_key = ?",
            (field_key,)
        ).fetchone()
    if row is None:
        return None
    if row["expires_at"] < now:
        return None
    return json.loads(row["results"])


def _set_catalog_cache(field_key, catalog, ra, dec, radius_arcsec, results):
    from parallax._db import get_db
    now = datetime.now(UTC)
    ttl = config.get("cache.catalog_ttl_days", 30)
    expires = now + timedelta(days=ttl)
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO catalog_cache "
            "(field_key, catalog, ra, dec, radius_arcsec, results, created_at, expires_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (field_key, catalog, ra, dec, radius_arcsec,
             json.dumps(results), now.isoformat(), expires.isoformat())
        )


def detect(
    fits_path: str,
    snr_threshold: float | None = None,
    min_pixels: int | None = None,
    filter_name: str | None = None,
) -> list[dict]:
    """Run source detection on a FITS image."""
    if not os.path.isfile(fits_path):
        raise FileNotFoundError(fits_path)

    if snr_threshold is None:
        snr_threshold = config.get("detection.snr_threshold", 3.0)
    if min_pixels is None:
        min_pixels = config.get("detection.min_pixels", 25)

    fwhm = config.get("detection.kernel_fwhm", 2.0)

    # cache key does not include filter_name
    h = _fits_hash(fits_path, snr_threshold, min_pixels, fwhm)

    if config.get("cache.detection_enabled"):
        cached = _get_detection_cache(h, snr_threshold, min_pixels, fwhm)
        if cached is not None:
            logger.info("detection cache hit for %s", os.path.basename(fits_path))
            if filter_name is not None:
                for s in cached:
                    s["filter"] = filter_name
            return cached

    from photutils.background import Background2D, MedianBackground
    from astropy.stats import SigmaClip
    from photutils.segmentation import detect_threshold, detect_sources, SourceCatalog
    from astropy.convolution import Gaussian2DKernel

    hdul = fits.open(fits_path)
    try:
        try:
            sci = _find_science_hdu(hdul)
        except (BufferError, TypeError, OSError) as exc:
            raise ValueError(
                f"Cannot read '{fits_path}' - file may be truncated or corrupt. "
                f"Delete it and re-run to trigger a fresh download."
            ) from exc
        if sci is None:
            raise ValueError("no valid 2D image HDU found")

        try:
            data = sci.data.astype(np.float64)
        except (BufferError, TypeError, OSError) as exc:
            raise ValueError(
                f"Cannot read '{fits_path}' - file may be truncated or corrupt. "
                f"Delete it and re-run to trigger a fresh download."
            ) from exc

        try:
            wcs = WCS(sci.header)
            if not wcs.has_celestial:
                wcs = None
                logger.warning("no celestial WCS in %s", fits_path)
        except Exception:
            wcs = None
            logger.warning("WCS extraction failed for %s", fits_path)

        pixar_sr = hdul[0].header.get("PIXAR_SR", None)
        if pixar_sr is None:
            pixar_sr = sci.header.get("PIXAR_SR", None)

        # load error and weight maps for flux uncertainty
        err_data = None
        try:
            err_data = hdul["ERR"].data.astype(np.float64)
        except (KeyError, AttributeError):
            pass

        wht_data = None
        try:
            wht_data = hdul["WHT"].data.astype(np.float64)
        except (KeyError, AttributeError):
            pass
    finally:
        hdul.close()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*nan.*", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        box_size = config.get("detection.background_box_size", 50)
        box_size_2 = config.get("detection.background_box_size_2", 0)
        filter_size = config.get("detection.background_filter_size", 3)
        interp_mode = config.get("detection.background_interp", "zoom")

        from photutils.background import BkgZoomInterpolator, BkgIDWInterpolator
        interp = BkgIDWInterpolator() if interp_mode == "idw" else BkgZoomInterpolator()

        try:
            bkg = Background2D(data, box_size=box_size,
                               filter_size=filter_size,
                               sigma_clip=SigmaClip(sigma=3),
                               bkg_estimator=MedianBackground(),
                               interpolator=interp)
        except Exception:
            fallback_box = max(4, min(box_size,
                                      data.shape[0] // 4,
                                      data.shape[1] // 4))
            try:
                bkg = Background2D(data, box_size=fallback_box,
                                   filter_size=filter_size,
                                   sigma_clip=SigmaClip(sigma=3),
                                   bkg_estimator=MedianBackground(),
                                   interpolator=interp)
                logger.debug("background fallback box_size=%d", fallback_box)
            except Exception:
                med = np.nanmedian(data)
                bkg = None

        two_scale = False
        if bkg is not None and box_size_2 > 0:
            try:
                bkg2 = Background2D(data, box_size=box_size_2,
                                    filter_size=filter_size,
                                    sigma_clip=SigmaClip(sigma=3),
                                    bkg_estimator=MedianBackground(),
                                    interpolator=interp)
                combined_bkg = np.minimum(bkg.background, bkg2.background)
                bkg_sub = data - combined_bkg
                bkg_rms = min(bkg.background_rms_median, bkg2.background_rms_median)
                two_scale = True
            except Exception:
                pass

        if not two_scale:
            if bkg is not None:
                bkg_sub = data - bkg.background
                bkg_rms = bkg.background_rms_median
            else:
                bkg_sub = data - med
                bkg_rms = max(np.std(data), 1e-10)

        rms_map = None
        if bkg is not None:
            rms_map = bkg.background_rms

        stddev = fwhm / 2.3548
        kernel = Gaussian2DKernel(stddev, x_size=5, y_size=5)
        convolved = convolve(bkg_sub, kernel)

        if two_scale:
            threshold = bkg_rms * snr_threshold
        elif bkg is not None:
            threshold = detect_threshold(bkg_sub, nsigma=snr_threshold,
                                         background=bkg.background)
        else:
            threshold = snr_threshold * bkg_rms

        segmap = detect_sources(convolved, threshold, npixels=min_pixels)
        if segmap is None:
            return []

        cat = SourceCatalog(bkg_sub, segmap, wcs=wcs)

        sources = []
        for src in cat:
            if wcs is not None:
                try:
                    sky = wcs.pixel_to_world(src.xcentroid, src.ycentroid)
                    ra_val = float(sky.ra.deg)
                    dec_val = float(sky.dec.deg)
                except Exception:
                    ra_val = float("nan")
                    dec_val = float("nan")
            else:
                ra_val = float("nan")
                dec_val = float("nan")

            peak = float(src.max_value)
            snr_val = peak / bkg_rms if bkg_rms > 0 else peak

            local_rms = None
            if rms_map is not None:
                try:
                    px = int(round(float(src.ycentroid)))
                    py = int(round(float(src.xcentroid)))
                    px = max(0, min(px, rms_map.shape[0] - 1))
                    py = max(0, min(py, rms_map.shape[1] - 1))
                    local_rms = float(rms_map[px, py])
                except Exception:
                    local_rms = None

            try:
                flux_val = float(src.kron_flux)
                if math.isnan(flux_val):
                    flux_val = float(src.segment_flux)
                    flux_src = "segment"
                else:
                    flux_src = "kron"
            except Exception:
                try:
                    flux_val = float(src.segment_flux)
                    flux_src = "segment"
                except Exception:
                    flux_val = 0.0
                    flux_src = "zero"

            flux_mjy_val = None
            mag_ab_val = None
            if pixar_sr is not None and pixar_sr > 0 and flux_val > 0:
                flux_mjy_val = flux_val * float(pixar_sr)
                try:
                    mag_ab_val = round(-2.5 * math.log10(flux_mjy_val / 3631.0), 4)
                except Exception:
                    mag_ab_val = None

            # flux uncertainty from ERR (and optionally WHT) extensions
            flux_err_val = None
            flux_mjy_err_val = None
            mag_ab_err_val = None
            if err_data is not None:
                try:
                    seg_mask = segmap.data == src.label
                    err_pixels = err_data[seg_mask]
                    if wht_data is not None:
                        w_pixels = wht_data[seg_mask]
                        # zero-weight pixels contribute zero variance
                        good = w_pixels > 0
                        if good.any():
                            var = np.zeros_like(err_pixels)
                            var[good] = (err_pixels[good] ** 2) / w_pixels[good]
                            flux_err_val = float(np.sqrt(np.nansum(var)))
                    else:
                        flux_err_val = float(np.sqrt(np.nansum(err_pixels ** 2)))

                    if flux_err_val is not None and flux_err_val > 0:
                        if pixar_sr is not None and pixar_sr > 0:
                            flux_mjy_err_val = flux_err_val * float(pixar_sr)
                        if flux_mjy_err_val and flux_mjy_val and flux_mjy_val > 0:
                            mag_ab_err_val = round(
                                2.5 / math.log(10) * flux_mjy_err_val / flux_mjy_val, 4)
                except Exception:
                    pass

            bb = src.bbox
            d = {
                "ra": ra_val,
                "dec": dec_val,
                "flux": flux_val,
                "snr": snr_val,
                "flux_source": flux_src,
                "pixel_x": float(src.xcentroid),
                "pixel_y": float(src.ycentroid),
                "label": int(src.label),
                "bbox": {
                    "ixmin": int(bb.ixmin),
                    "ixmax": int(bb.ixmax),
                    "iymin": int(bb.iymin),
                    "iymax": int(bb.iymax),
                },
                "flux_mjy": flux_mjy_val,
                "mag_ab": mag_ab_val,
                "flux_err": flux_err_val,
                "flux_mjy_err": flux_mjy_err_val,
                "mag_ab_err": mag_ab_err_val,
                "local_rms": local_rms,
                "field_rms": bkg_rms,
            }
            if filter_name is not None:
                d["filter"] = filter_name
            sources.append(d)

    if config.get("cache.detection_enabled"):
        # store without filter label so cache is reusable
        cache_sources = [{k: v for k, v in s.items() if k != "filter"} for s in sources]
        _set_detection_cache(fits_path, h, snr_threshold, min_pixels, fwhm, cache_sources)

    return sources


def _merge_detections(all_detections: list[dict], match_radius_arcsec: float = 2.0) -> list[dict]:
    if not all_detections:
        return []

    valid = []
    nan_dets = []
    for d in all_detections:
        if math.isnan(d["ra"]) or math.isnan(d["dec"]):
            filt = d.get("filter", "UNKNOWN")
            entry = dict(d)
            entry["detections"] = [{"filter": filt, "flux": d["flux"],
                                     "snr": d["snr"], "pixel_x": d["pixel_x"],
                                     "pixel_y": d["pixel_y"],
                                     "flux_source": d.get("flux_source", "kron"),
                                     "flux_mjy": d.get("flux_mjy"),
                                     "mag_ab": d.get("mag_ab"),
                                     "flux_err": d.get("flux_err"),
                                     "flux_mjy_err": d.get("flux_mjy_err"),
                                     "mag_ab_err": d.get("mag_ab_err")}]
            nan_dets.append(entry)
        else:
            valid.append(d)

    if not valid:
        return nan_dets

    ras = np.array([d["ra"] for d in valid])
    decs = np.array([d["dec"] for d in valid])
    coords = SkyCoord(ras, decs, unit="deg")

    # greedy clustering: assign each detection to a group
    used = np.zeros(len(valid), dtype=bool)
    groups = []  # each group is a list of indices

    # sort by SNR descending so highest-SNR sources become group leaders
    snr_order = np.argsort([-d["snr"] for d in valid])

    for idx in snr_order:
        if used[idx]:
            continue
        # find all unmatched detections within radius
        seps = coords[idx].separation(coords)
        nearby = np.where((seps.arcsec <= match_radius_arcsec) & (~used))[0]
        for n in nearby:
            used[n] = True
        groups.append(list(nearby))

    merged = []
    for group in groups:
        dets_in_group = [valid[i] for i in group]
        # best by SNR
        best = max(dets_in_group, key=lambda x: x["snr"])
        entry = dict(best)
        entry["detections"] = []
        for d in dets_in_group:
            filt = d.get("filter", "UNKNOWN")
            entry["detections"].append({
                "filter": filt, "flux": d["flux"], "snr": d["snr"],
                "pixel_x": d["pixel_x"], "pixel_y": d["pixel_y"],
                "flux_source": d.get("flux_source", "kron"),
                "flux_mjy": d.get("flux_mjy"), "mag_ab": d.get("mag_ab"),
                "flux_err": d.get("flux_err"), "flux_mjy_err": d.get("flux_mjy_err"),
                "mag_ab_err": d.get("mag_ab_err"),
            })
        merged.append(entry)

    return merged + nan_dets


def _query_simbad(center_ra, center_dec, radius_deg, timeout) -> list[dict]:
    radius_arcsec = radius_deg * 3600.0
    if config.get("cache.catalog_enabled"):
        key = f"{center_ra:.4f}_{center_dec:.4f}_{radius_arcsec}_SIMBAD"
        cached = _get_catalog_cache(key)
        if cached is not None:
            logger.debug("catalog cache HIT: %s", key)
            return cached
        logger.debug("catalog cache MISS: %s", key)

    from astroquery.simbad import Simbad
    try:
        result = Simbad.query_tap(
            f"SELECT main_id, ra, dec, otype, rvz_redshift "
            f"FROM basic WHERE CONTAINS("
            f"POINT('ICRS', ra, dec), "
            f"CIRCLE('ICRS', {center_ra}, {center_dec}, {radius_deg})) = 1"
        )
    except Exception as e:
        logger.warning("SIMBAD query failed: %s", e)
        return []

    if result is None or len(result) == 0:
        return []

    rows = []
    for row in result:
        rz = row["rvz_redshift"]
        rows.append({
            "catalog": "SIMBAD",
            "source_id": str(row["main_id"]),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "object_type": str(row["otype"]) if row["otype"] else None,
            "redshift": float(rz) if rz is not None and not np.ma.is_masked(rz) else None,
        })
    if config.get("cache.catalog_enabled"):
        _set_catalog_cache(key, "SIMBAD", center_ra, center_dec, radius_arcsec, rows)
    return rows


def _query_ned(center_ra, center_dec, radius_arcsec, timeout) -> list[dict]:
    if config.get("cache.catalog_enabled"):
        key = f"{center_ra:.4f}_{center_dec:.4f}_{radius_arcsec}_NED"
        cached = _get_catalog_cache(key)
        if cached is not None:
            logger.debug("catalog cache HIT: %s", key)
            return cached
        logger.debug("catalog cache MISS: %s", key)

    from astroquery.ipac.ned import Ned
    try:
        Ned.TIMEOUT = timeout
        coord = SkyCoord(center_ra, center_dec, unit="deg")
        result = Ned.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        logger.warning("NED query failed: %s", e)
        return []

    if result is None or len(result) == 0:
        return []

    rows = []
    for row in result:
        try:
            obj_ra = float(row["RA"])
            obj_dec = float(row["DEC"])
        except (KeyError, ValueError):
            obj_ra = float(row["ra"])
            obj_dec = float(row["dec"])
        if np.ma.is_masked(row.get("Redshift", None)):
            rz = None
        else:
            try:
                rz = float(row["Redshift"])
            except Exception:
                rz = None
        rows.append({
            "catalog": "NED",
            "source_id": str(row["Object Name"]) if "Object Name" in row.colnames else "",
            "ra": obj_ra,
            "dec": obj_dec,
            "object_type": str(row["Type"]) if "Type" in row.colnames and not np.ma.is_masked(row["Type"]) else None,
            "redshift": rz,
        })
    if config.get("cache.catalog_enabled"):
        _set_catalog_cache(key, "NED", center_ra, center_dec, radius_arcsec, rows)
    return rows


def _query_gaia(center_ra, center_dec, radius_arcsec, timeout) -> list[dict]:
    if config.get("cache.catalog_enabled"):
        key = f"{center_ra:.4f}_{center_dec:.4f}_{radius_arcsec}_GAIA"
        cached = _get_catalog_cache(key)
        if cached is not None:
            logger.debug("catalog cache HIT: %s", key)
            return cached
        logger.debug("catalog cache MISS: %s", key)

    from astroquery.gaia import Gaia
    warnings.filterwarnings("ignore", message=".*archive is unstable.*")
    gaia_logger = logging.getLogger("astroquery.gaia")
    prev_level = gaia_logger.level
    gaia_logger.setLevel(logging.CRITICAL)
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    try:
        coord = SkyCoord(center_ra, center_dec, unit="deg")
        # gaia server prints "archive is unstable" directly to stdout
        _real_stdout = sys.stdout
        _devnull = None
        try:
            try:
                _devnull = open(os.devnull, "w")
                sys.stdout = _devnull
            except Exception:
                pass
            job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
            result = job.get_results()
        finally:
            sys.stdout = _real_stdout
            if _devnull is not None:
                try:
                    _devnull.close()
                except Exception:
                    pass
    except KeyboardInterrupt:
        raise
    except Exception as e:
        logger.warning("Gaia query failed: %s", e)
        return []
    finally:
        gaia_logger.setLevel(prev_level or logging.WARNING)

    if result is None or len(result) == 0:
        return []

    rows = []
    for row in result:
        rows.append({
            "catalog": "GAIA",
            "source_id": str(row["source_id"]),
            "ra": float(row["ra"]),
            "dec": float(row["dec"]),
            "object_type": None,
            "redshift": None,
        })
    if config.get("cache.catalog_enabled"):
        _set_catalog_cache(key, "GAIA", center_ra, center_dec, radius_arcsec, rows)
    return rows


def _compute_field_circle(detections, search_radius_arcsec):
    valid = [(d["ra"], d["dec"]) for d in detections
             if not (math.isnan(d["ra"]) or math.isnan(d["dec"]))]
    if not valid:
        return None, None, None

    ras = [v[0] for v in valid]
    decs = [v[1] for v in valid]
    center_ra = np.mean(ras)
    center_dec = np.mean(decs)

    center = SkyCoord(center_ra, center_dec, unit="deg")
    corners = SkyCoord(ras, decs, unit="deg")
    seps = center.separation(corners)
    max_sep_arcsec = float(seps.max().arcsec) if len(seps) > 0 else 0.0

    field_radius_arcsec = max_sep_arcsec + search_radius_arcsec
    return float(center_ra), float(center_dec), field_radius_arcsec


def _bbox_area(det):
    bb = det.get("bbox")
    if not bb:
        return None
    try:
        return (bb["ixmax"] - bb["ixmin"]) * (bb["iymax"] - bb["iymin"])
    except (KeyError, TypeError):
        return None


def _is_narrowband(filter_name):
    f = filter_name.strip().upper()
    return f.endswith("N")


def _compute_confidence(snr, n_filters, total_filters, nearest_sep_arcsec, flux_source,
                        narrowband_only=False):
    search_radius = 2.0  # match resolver default
    snr_score = min(snr, 30.0) / 30.0
    filter_score = n_filters / max(total_filters, 1)
    if nearest_sep_arcsec == float("inf"):
        near_miss_score = 1.0
    elif nearest_sep_arcsec <= search_radius:
        near_miss_score = 0.0
    else:
        near_miss_score = min(1.0, (nearest_sep_arcsec - search_radius) / (10.0 - search_radius))
    flux_score = {"kron": 1.0, "segment": 0.5, "zero": 0.0}.get(flux_source, 1.0)
    raw = 0.30 * snr_score + 0.35 * filter_score + 0.20 * near_miss_score + 0.15 * flux_score
    if narrowband_only:
        raw *= 0.85
    return round(max(0.0, min(1.0, raw)), 4)


def _cross_match_local(detections, catalog_rows, search_radius_arcsec):
    n = len(detections)
    if not catalog_rows:
        return [[] for _ in range(n)], [float("inf")] * n

    from astropy.coordinates import match_coordinates_sky

    cat_ra = np.array([r["ra"] for r in catalog_rows])
    cat_dec = np.array([r["dec"] for r in catalog_rows])
    cat_coords = SkyCoord(cat_ra, cat_dec, unit="deg")

    det_ra = np.array([d["ra"] for d in detections])
    det_dec = np.array([d["dec"] for d in detections])
    det_coords = SkyCoord(det_ra, det_dec, unit="deg")

    idx, sep2d, _ = det_coords.match_to_catalog_sky(cat_coords)

    matches = []
    nearest_seps = []
    for i in range(n):
        nearest_seps.append(float(sep2d[i].arcsec))
        if sep2d[i].arcsec <= search_radius_arcsec:
            row = catalog_rows[idx[i]]
            matches.append([CatalogMatch(
                catalog=row["catalog"],
                source_id=row["source_id"],
                separation_arcsec=float(sep2d[i].arcsec),
                object_type=row.get("object_type"),
                redshift=row.get("redshift"),
            )])
        else:
            matches.append([])
    return matches, nearest_seps


def resolve(
    detections: list[dict],
    search_radius: float | None = None,
) -> tuple[list[Candidate], bool]:
    """Cross-reference detections against SIMBAD, NED, and Gaia."""
    if search_radius is None:
        search_radius = config.get("resolver.search_radius_arcsec", 2.0)
    timeout = config.get("resolver.timeout_seconds", 30)

    valid_idx = [i for i, d in enumerate(detections)
                 if not (math.isnan(d["ra"]) or math.isnan(d["dec"]))]
    valid_dets = [detections[i] for i in valid_idx]

    catalog_results = {}

    all_failed = True
    gaia_failed = False

    if valid_dets:
        fc_ra, fc_dec, field_radius_arcsec = _compute_field_circle(detections, search_radius)
        field_radius_deg = field_radius_arcsec / 3600.0

        for name, qfn, radius_arg in [
            ("SIMBAD", _query_simbad, field_radius_deg),
            ("NED", _query_ned, field_radius_arcsec),
            ("GAIA", _query_gaia, field_radius_arcsec),
        ]:
            try:
                rows = qfn(fc_ra, fc_dec, radius_arg, timeout)
                catalog_results[name] = rows
                all_failed = False
            except Exception as e:
                if name == "GAIA":
                    gaia_failed = True
                logger.warning("catalog query failed in resolve: %s", e)
    else:
        all_failed = False

    per_det_matches = [[] for _ in valid_dets]
    nearest_catalog_sep = [float("inf")] * len(valid_dets)
    if valid_dets:
        for cat_name, rows in catalog_results.items():
            matched, nearest_seps = _cross_match_local(valid_dets, rows, search_radius)
            for i, m in enumerate(matched):
                per_det_matches[i].extend(m)
                if nearest_seps[i] < nearest_catalog_sep[i]:
                    nearest_catalog_sep[i] = nearest_seps[i]

    match_by_idx = {}
    sep_by_idx = {}
    for vi, oi in enumerate(valid_idx):
        match_by_idx[oi] = per_det_matches[vi]
        sep_by_idx[oi] = nearest_catalog_sep[vi]

    match_radius = config.get("cache.candidate_match_radius_arcsec", 2.0)
    dedup_map = {}  # detection index -> existing Candidate
    from parallax import catalog as _cat

    if valid_dets and fc_ra is not None:
        existing_cands = _cat.query(fc_ra, fc_dec, field_radius_arcsec)
        if existing_cands:
            ex_ra = np.array([c.ra for c in existing_cands])
            ex_dec = np.array([c.dec for c in existing_cands])
            ex_coords = SkyCoord(ex_ra, ex_dec, unit="deg")

            det_ra = np.array([detections[i]["ra"] for i in valid_idx])
            det_dec = np.array([detections[i]["dec"] for i in valid_idx])
            det_coords = SkyCoord(det_ra, det_dec, unit="deg")

            idx, sep2d, _ = det_coords.match_to_catalog_sky(ex_coords)
            for vi, oi in enumerate(valid_idx):
                if sep2d[vi].arcsec <= match_radius:
                    dedup_map[oi] = existing_cands[idx[vi]]

    _cls_rank = {"unverified": 0, "known": 1}

    total_filters = max(1, len(set(
        d.get("filter", "UNKNOWN")
        for det in detections
        for d in det.get("detections", [{"filter": det.get("filter", "UNKNOWN")}])
    )))

    candidates = []
    for i, det in enumerate(detections):
        det_matches = match_by_idx.get(i, [])

        catalogs_with_hits = set()
        has_redshift = False
        for m in det_matches:
            catalogs_with_hits.add(m.catalog)
            if m.redshift is not None:
                has_redshift = True

        n_cats = len(catalogs_with_hits)
        if n_cats >= 1 or has_redshift:
            cls = "known"
        else:
            cls = "unverified"

        # build Detection list from merged detections if present
        cand_detections = []
        if "detections" in det:
            for dd in det["detections"]:
                cand_detections.append(Detection(
                    filter=dd.get("filter", "UNKNOWN"),
                    flux=dd["flux"], snr=dd["snr"],
                    pixel_coords=(dd["pixel_x"], dd["pixel_y"]),
                    flux_mjy=dd.get("flux_mjy"),
                    mag_ab=dd.get("mag_ab"),
                    flux_err=dd.get("flux_err"),
                    flux_mjy_err=dd.get("flux_mjy_err"),
                    mag_ab_err=dd.get("mag_ab_err"),
                ))

        # narrowband analysis
        nb_only = False
        line_dominated = False
        if cand_detections:
            nb_flags = [_is_narrowband(d.filter) for d in cand_detections]
            if all(nb_flags):
                nb_only = True
            elif any(nb_flags):
                nb_snrs = [d.snr for d, nb in zip(cand_detections, nb_flags) if nb]
                bb_snrs = [d.snr for d, nb in zip(cand_detections, nb_flags) if not nb]
                if bb_snrs and max(nb_snrs) >= 2.0 * max(bb_snrs):
                    line_dominated = True

        # morphology tags from bbox
        area = _bbox_area(det)
        compact = area is not None and area <= 64
        extended = area is not None and area >= 400

        # proximity tags from catalog separation
        nsep = sep_by_idx.get(i, float("inf"))
        isolated = nsep == float("inf") or nsep > 30.0
        crowded = nsep > search_radius and nsep <= 10.0

        # background structure tag
        near_emission = False
        local_rms = det.get("local_rms")
        field_rms = det.get("field_rms")
        if (local_rms is not None and field_rms is not None
                and field_rms > 0 and local_rms > field_rms * 2.0):
            near_emission = True

        auto_tags = []
        if nb_only:
            auto_tags.append("narrowband_only")
        if line_dominated:
            auto_tags.append("line_dominated")
        if compact:
            auto_tags.append("compact")
        if extended:
            auto_tags.append("extended")
        if isolated:
            auto_tags.append("isolated")
        if crowded:
            auto_tags.append("crowded")
        if near_emission:
            auto_tags.append("near_emission")

        # confidence inputs
        n_filt = max(1, len(set(
            dd.get("filter", "UNKNOWN") for dd in det.get("detections", [{"filter": det.get("filter", "UNKNOWN")}])
        )))
        fsrc = det.get("flux_source", "kron")
        conf = _compute_confidence(det["snr"], n_filt, total_filters, nsep, fsrc,
                                   narrowband_only=nb_only)

        # pick best-SNR detection that has uncertainty
        _best_err = None
        for _d in sorted(cand_detections, key=lambda d: d.snr, reverse=True):
            if _d.flux_err is not None:
                _best_err = _d
                break

        ex = dedup_map.get(i)
        if ex is not None:
            if _cls_rank.get(cls, 0) > _cls_rank.get(ex.classification, 0):
                _cat.update(ex.id, classification=cls)
                ex.classification = cls
            ex.catalog_matches = det_matches
            ex.detections = cand_detections
            ex.confidence = conf
            ex.flux_err = _best_err.flux_err if _best_err else None
            ex.flux_mjy_err = _best_err.flux_mjy_err if _best_err else None
            ex.mag_ab_err = _best_err.mag_ab_err if _best_err else None
            merged_tags = list(ex.tags)
            for t in auto_tags:
                if t not in merged_tags:
                    merged_tags.append(t)
            ex.tags = merged_tags
            _cat.update(ex.id, tags=merged_tags)
            candidates.append(ex)
        else:
            cand = Candidate(
                id=_candidate_id(),
                ra=det["ra"],
                dec=det["dec"],
                flux=det["flux"],
                snr=det["snr"],
                classification=cls,
                report_id="",
                pixel_coords=(det["pixel_x"], det["pixel_y"]),
                created_at=datetime.now(UTC),
                catalog_matches=det_matches,
                detections=cand_detections,
                confidence=conf,
                tags=auto_tags,
                flux_err=_best_err.flux_err if _best_err else None,
                flux_mjy_err=_best_err.flux_mjy_err if _best_err else None,
                mag_ab_err=_best_err.mag_ab_err if _best_err else None,
            )
            candidates.append(cand)

    if all_failed and detections:
        has_coords = len(valid_idx) > 0
        if has_coords:
            raise ConnectionError("all catalog queries failed")

    return candidates, gaia_failed


def report(
    candidates: list[Candidate],
    target: str,
    fits_inputs: list[tuple[str, str]],
    n_sources_detected: int,
    output_format: str | None = None,
    gaia_failed: bool = False,
) -> Report:
    """Build a Report from resolved candidates and write it to disk."""
    if output_format is None:
        output_format = config.get("report.output_format", "both")

    snr = config.get("detection.snr_threshold", 3.0)
    npix = config.get("detection.min_pixels", 25)
    fwhm = config.get("detection.kernel_fwhm", 2.0)
    try:
        fp = _run_fingerprint(target, fits_inputs, snr, npix, fwhm)
    except OSError:
        fp = None

    first_path = fits_inputs[0][0]
    with fits.open(first_path) as hdul:
        sci = _find_science_hdu(hdul)
        hdr = sci.header if sci else hdul[0].header
        instrument = hdr.get("INSTRUME", hdul[0].header.get("INSTRUME", "UNKNOWN"))

    filter_list = list(dict.fromkeys(f for _, f in fits_inputs))
    obs_ids = []
    for path, _ in fits_inputs:
        with fits.open(path) as hdul:
            primary_hdr = hdul[0].header
            obs_id = primary_hdr.get("OBS_ID", primary_hdr.get("OBSERVTN", "UNKNOWN"))
            obs_ids.append(obs_id)

    rpt_id = _report_id(target, fp)
    now = datetime.now(UTC)

    for c in candidates:
        c.report_id = rpt_id

    n_matched = sum(1 for c in candidates
                    if c.classification == "known")
    n_unverified = sum(1 for c in candidates
                       if c.classification == "unverified")

    reports_dir = os.path.join(
        config.get("data.reports_path"),
        _target_slug(target),
        now.strftime("%Y%m%d"),
    )
    os.makedirs(reports_dir, exist_ok=True)
    json_path = None
    md_path = None

    if output_format in ("json", "both"):
        json_path = os.path.normpath(os.path.join(reports_dir, f"{rpt_id}.json"))
    if output_format in ("markdown", "both"):
        md_path = os.path.normpath(os.path.join(reports_dir, f"{rpt_id}.md"))

    rpt = Report(
        id=rpt_id,
        target=target,
        instrument=instrument,
        filters=filter_list,
        created_at=now,
        candidates=candidates,
        n_sources_detected=n_sources_detected,
        n_catalog_matched=n_matched,
        n_unverified=n_unverified,
        json_path=json_path,
        md_path=md_path,
    )

    include_known = config.get("report.include_known", False)

    if json_path:
        d = report_to_dict(rpt)
        with open(json_path, "w") as f:
            json.dump(d, f, indent=2, default=str)

    if md_path:
        _write_markdown(rpt, md_path, include_known, gaia_failed,
                        fits_inputs=fits_inputs, obs_ids=obs_ids)

        import parallax as _par
        prov_path = os.path.join(reports_dir, f"{rpt_id}_provenance.json")
        prov = {
            "parallax_version": _par.__version__,
            "report_id": rpt_id,
            "target": target,
            "created_at": now.isoformat(),
            "instrument": instrument,
            "filters": filter_list,
            "catalogs_queried": config.get("resolver.catalogs", []),
            "gaia_available": not gaia_failed,
            "pipeline": {
                "snr_threshold": config.get("detection.snr_threshold", 3.0),
                "min_pixels": config.get("detection.min_pixels", 25),
                "kernel_fwhm": config.get("detection.kernel_fwhm", 2.0),
                "background_box_size": config.get("detection.background_box_size", 50),
                "search_radius_arcsec": config.get("resolver.search_radius_arcsec", 2.0),
                "narrowband_only_count": sum(1 for c in candidates if "narrowband_only" in c.tags),
                "line_dominated_count": sum(1 for c in candidates if "line_dominated" in c.tags),
                "compact_count": sum(1 for c in candidates if "compact" in c.tags),
                "extended_count": sum(1 for c in candidates if "extended" in c.tags),
                "isolated_count": sum(1 for c in candidates if "isolated" in c.tags),
                "crowded_count": sum(1 for c in candidates if "crowded" in c.tags),
                "near_emission_count": sum(1 for c in candidates if "near_emission" in c.tags),
            },
            "flux_calibration": "MJy via PIXAR_SR header keyword, uncertainties from ERR/WHT extensions, no aperture correction",
            "input_files": [
                {"obs_id": oid, "filter": filt, "filename": os.path.basename(path)}
                for (path, filt), oid in zip(fits_inputs, obs_ids)
            ],
        }
        with open(prov_path, "w") as f:
            json.dump(prov, f, indent=2, default=str)

    # TODO: batch the db writes
    from parallax._db import get_db
    with get_db() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO reports (id, target, instrument, filter, observation_id, "
            "fits_path, created_at, n_sources_detected, n_catalog_matched, "
            "n_unverified, json_path, md_path, fingerprint, filters) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rpt.id, rpt.target, rpt.instrument,
             filter_list[0] if filter_list else None,
             obs_ids[0] if obs_ids else None,
             first_path,
             rpt.created_at.isoformat(), rpt.n_sources_detected,
             rpt.n_catalog_matched, rpt.n_unverified, rpt.json_path, rpt.md_path,
             fp, json.dumps(filter_list))
        )

        for (path, filt), obs in zip(fits_inputs, obs_ids):
            conn.execute(
                "INSERT INTO report_inputs (report_id, fits_path, observation_id, filter) "
                "VALUES (?,?,?,?)",
                (rpt.id, path, obs, filt)
            )

    from parallax import catalog
    n_persisted = 0
    for c in rpt.candidates:
        try:
            catalog.add(c)
            n_persisted += 1
        except Exception as e:
            logger.debug("could not persist candidate %s: %s", c.id, e)
    if n_persisted < len(rpt.candidates):
        logger.info("persisted %d/%d candidates", n_persisted, len(rpt.candidates))

    return rpt


def _write_markdown(rpt, path, include_known, gaia_failed=False,
                    fits_inputs=None, obs_ids=None):
    lines = [
        f"# Parallax Report: {rpt.target}",
        f"{rpt.created_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        f"Instrument: {rpt.instrument} / {' '.join(rpt.filters)}",
        "",
        "## Summary",
        f"Sources detected: {rpt.n_sources_detected}",
        f"Catalog matched: {rpt.n_catalog_matched}",
        f"Unverified: {rpt.n_unverified}",
    ]

    if rpt.filters and len(rpt.filters) > 1:
        filter_only = {}
        for c in rpt.candidates:
            if c.detections and len(c.detections) == 1:
                f = c.detections[0].filter
                filter_only[f] = filter_only.get(f, 0) + 1
        if filter_only:
            parts = [f"{f}: {n}" for f, n in sorted(filter_only.items())]
            lines.append(f"Filter-unique sources: {', '.join(parts)}")

    _seen_ids = set()
    _deduped = []
    for c in rpt.candidates:
        if c.id not in _seen_ids:
            _seen_ids.add(c.id)
            _deduped.append(c)
    n_nb = sum(1 for c in _deduped if "narrowband_only" in c.tags)
    n_ld = sum(1 for c in _deduped if "line_dominated" in c.tags)
    n_compact = sum(1 for c in _deduped if "compact" in c.tags)
    n_extended = sum(1 for c in _deduped if "extended" in c.tags)
    n_isolated = sum(1 for c in _deduped if "isolated" in c.tags)
    n_crowded = sum(1 for c in _deduped if "crowded" in c.tags)
    n_ne = sum(1 for c in _deduped if "near_emission" in c.tags)
    if n_nb:
        lines.append(f"Narrowband-only: {n_nb}")
    if n_ld:
        lines.append(f"Line-dominated: {n_ld}")
    if n_compact:
        lines.append(f"Compact sources: {n_compact}")
    if n_extended:
        lines.append(f"Extended sources: {n_extended}")
    if n_isolated:
        lines.append(f'Isolated (>30"): {n_isolated}')
    if n_crowded:
        lines.append(f'Crowded (<10"): {n_crowded}')
    if n_ne:
        lines.append(f"Near-emission region: {n_ne}")

    if gaia_failed:
        lines.append("Gaia: unavailable during this run")

    lines.append("")
    lines.append("Confidence scores reflect detection quality: SNR, number of filters in")
    lines.append("which the source was detected, distance to the nearest catalog source, and")
    lines.append("flux measurement reliability. A higher score means the detection is better")
    lines.append("measured, not that the source is more likely to be real.")
    lines.append("")

    # dedup by candidate id - resolve() can produce duplicates via dedup merging
    seen = set()
    unverified = []
    for c in rpt.candidates:
        if c.id in seen:
            continue
        seen.add(c.id)
        if c.classification == "unverified":
            unverified.append(c)

    md_limit = 100

    unverified_by_snr = sorted(unverified, key=lambda c: c.snr, reverse=True)
    lines.append("## Unverified Candidates")
    lines.append("| ID | RA | Dec | SNR | Confidence | Flux | Flux(MJy) | Flux err | Mag(AB) | Mag err |")
    lines.append("|----|----|-----|-----|------------|------|-----------|----------|---------|---------|")
    for c in unverified_by_snr[:md_limit]:
        best_cal = None
        if c.detections:
            for det in sorted(c.detections, key=lambda d: d.snr, reverse=True):
                if det.flux_mjy is not None:
                    best_cal = det
                    break
        fmjy = f"{best_cal.flux_mjy:.4e}" if best_cal and best_cal.flux_mjy is not None else "-"
        ferr = f"{best_cal.flux_mjy_err:.4e}" if best_cal and best_cal.flux_mjy_err is not None else "-"
        mab = f"{best_cal.mag_ab:.2f}" if best_cal and best_cal.mag_ab is not None else "-"
        merr = f"{best_cal.mag_ab_err:.4f}" if best_cal and best_cal.mag_ab_err is not None else "-"
        lines.append(f"| {c.id} | {c.ra:.4f} | {c.dec:.4f} | {c.snr:.1f} | {c.confidence:.2f} | {c.flux:.1f} | {fmjy} | {ferr} | {mab} | {merr} |")
    if len(unverified) > md_limit:
        lines.append(f"\n... and {len(unverified) - md_limit} more unverified candidates (see JSON for full list)")

    if include_known:
        lines.append("")
        lines.append("## Known")
        known = []
        for c in rpt.candidates:
            if c.classification == "known" and c.id not in seen:
                seen.add(c.id)
                known.append(c)
        known_by_snr = sorted(known, key=lambda c: c.snr, reverse=True)
        lines.append("| ID | RA | Dec | SNR | Catalog |")
        lines.append("|----|----|-----|-----|---------|")
        for c in known_by_snr[:md_limit]:
            cats = ", ".join(m.catalog for m in c.catalog_matches)
            lines.append(f"| {c.id} | {c.ra:.4f} | {c.dec:.4f} | {c.snr:.1f} | {cats} |")
        if len(known) > md_limit:
            lines.append(f"\n... and {len(known) - md_limit} more known candidates (see JSON for full list)")

    lines.append("")
    lines.append("## Caveats")
    lines.append("Flux(MJy) is Kron aperture photometry converted from MJy/sr using the")
    lines.append("PIXAR_SR header keyword. Flux uncertainties are propagated from the i2d")
    lines.append("ERR extension (weighted by WHT where available). No aperture correction")
    lines.append("has been applied.")
    lines.append("Point source fluxes may be underestimated by 10-20% depending on filter")
    lines.append("and source size. Extended source fluxes will be underestimated further.")
    lines.append("Background subtraction in structured emission fields may include nebular")
    lines.append("knots as discrete sources. Auto-tags (narrowband_only, line_dominated,")
    lines.append("compact, extended, isolated, crowded) are heuristic flags, not physical")
    lines.append("measurements. Confidence scores reflect detection quality, not")
    lines.append("astrophysical significance.")

    # provenance
    import parallax as _par
    lines.append("")
    lines.append("## Provenance")
    lines.append(f"Parallax: {_par.__version__}")
    lines.append(f"Run: {rpt.created_at.isoformat()}")
    lines.append(f"Target: {rpt.target}")
    lines.append(f"Instrument: {rpt.instrument}")
    lines.append(f"Filters: {' '.join(rpt.filters)}")
    lines.append(f"Catalogs: {', '.join(config.get('resolver.catalogs', []))}")
    lines.append(f"SNR threshold: {config.get('detection.snr_threshold', 3.0)}")
    lines.append(f"Min pixels: {config.get('detection.min_pixels', 25)}")
    lines.append(f"Kernel FWHM: {config.get('detection.kernel_fwhm', 2.0)}")
    lines.append(f"Background box size: {config.get('detection.background_box_size', 50)}")
    lines.append(f"Search radius: {config.get('resolver.search_radius_arcsec', 2.0)} arcsec")
    lines.append(f"Gaia: {'unavailable' if gaia_failed else 'available'}")

    if fits_inputs and obs_ids:
        lines.append("")
        lines.append("Input files:")
        lines.append("| Obs ID | Filter | Filename |")
        lines.append("|--------|--------|----------|")
        for (fpath, filt), oid in zip(fits_inputs, obs_ids):
            lines.append(f"| {oid} | {filt} | {os.path.basename(fpath)} |")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _run_fingerprint(target, fits_inputs, snr_threshold, min_pixels, fwhm):
    parts = [target]
    for path, filt in sorted(fits_inputs):
        mtime = os.path.getmtime(path)
        parts.append(f"{path}:{mtime}:{filt}")
    parts += [str(snr_threshold), str(min_pixels), str(fwhm)]
    key = ":".join(parts)
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _fmt_ra(ra):
    h = int(ra / 15)
    m = int((ra / 15 - h) * 60)
    s = int(((ra / 15 - h) * 60 - m) * 60)
    return f"{h:02d}{m:02d}{s:02d}"


def _fmt_dec(dec):
    sign = "+" if dec >= 0 else "-"
    d = int(abs(dec))
    m = int((abs(dec) - d) * 60)
    s = int(((abs(dec) - d) * 60 - m) * 60)
    return f"{sign}{d:02d}{m:02d}{s:02d}"


def reduce(
    target: str | None = None,
    instrument: str | None = None,
    filters: list[str] | None = None,
    on_progress: callable | None = None,
    ra: float | None = None,
    dec: float | None = None,
) -> Report:
    """Full pipeline: acquire -> detect -> resolve -> report."""
    if target is None and (ra is None or dec is None):
        raise ValueError("target or ra/dec required")
    if target is None:
        target = f"J{_fmt_ra(ra)}{_fmt_dec(dec)}"

    fits_paths = acquire(target, instrument, filters, on_progress=on_progress,
                         ra=ra, dec=dec)
    if on_progress:
        on_progress("acquire", f"{len(fits_paths)} file(s) found")

    all_detections = []
    fits_inputs = []
    for p in fits_paths:
        with fits.open(p) as hdul:
            # filter name lives in primary HDU on real JWST i2d files
            primary_hdr = hdul[0].header
            filt = primary_hdr.get("FILTER", primary_hdr.get("FILTER1"))
            if not filt and _find_science_hdu(hdul) is not None:
                sci_hdr = _find_science_hdu(hdul).header
                filt = sci_hdr.get("FILTER", sci_hdr.get("FILTER1", "UNKNOWN"))
            if not filt:
                filt = "UNKNOWN"
        fits_inputs.append((p, filt))
        dets = detect(p, filter_name=filt)
        all_detections.extend(dets)
        if on_progress:
            on_progress("detect", f"{filt}: {len(dets)} sources")

    merged = _merge_detections(all_detections)
    if on_progress:
        on_progress("merge", f"{len(merged)} merged sources")

    n_total = len(merged)
    candidates, gaia_flag = resolve(merged)
    if on_progress:
        on_progress("resolve", f"{len(candidates)} candidates")

    rpt = report(candidates, target, fits_inputs, n_total, gaia_failed=gaia_flag)
    if on_progress:
        on_progress("report", rpt.id)

    report_dir = os.path.dirname(rpt.md_path) if rpt.md_path else None
    if report_dir:
        import parallax.chart as _chart
        chart_path = os.path.normpath(os.path.join(report_dir, f"{rpt.id}_chart.png"))
        try:
            _chart.plot(rpt, output_path=chart_path)
            if on_progress:
                on_progress("chart", chart_path)
        except Exception as e:
            logger.warning("chart auto-save failed: %s", e)

        import parallax.view as _view
        try:
            unverified = sorted(
                (c for c in rpt.candidates if c.classification == "unverified"),
                key=lambda c: c.confidence, reverse=True,
            )
            cutout_path = os.path.normpath(os.path.join(report_dir, f"{rpt.id}_cutout.png"))
            session = _view.open(rpt)
            for cand in unverified[:10]:
                try:
                    cv = _view.examine(cand, session)
                    _view.show(cv, output_path=cutout_path)
                    if on_progress:
                        on_progress("cutout", cutout_path)
                    break
                except Exception:
                    continue
        except Exception as e:
            logger.warning("cutout auto-save failed: %s", e)

    return rpt


def cache_status() -> dict:
    """Return current cache state."""
    from parallax._db import get_db
    result = {"detection": [], "catalog": []}
    with get_db() as conn:
        for row in conn.execute("SELECT * FROM detection_cache").fetchall():
            dets = json.loads(row["detections"])
            result["detection"].append({
                "fits_path": row["fits_path"],
                "fits_hash": row["fits_hash"],
                "detections_count": len(dets),
                "cached_at": row["created_at"],
                "params": {
                    "snr_threshold": row["snr_threshold"],
                    "min_pixels": row["min_pixels"],
                    "kernel_fwhm": row["kernel_fwhm"],
                },
            })
        for row in conn.execute("SELECT * FROM catalog_cache").fetchall():
            res = json.loads(row["results"])
            result["catalog"].append({
                "field_key": row["field_key"],
                "catalog": row["catalog"],
                "ra": row["ra"],
                "dec": row["dec"],
                "cached_at": row["created_at"],
                "expires_at": row["expires_at"],
                "result_count": len(res),
            })
    return result


def clear_cache(fits_path: str | None = None) -> dict:
    """Clear cache entries. If fits_path given, only detection cache for that file."""
    from parallax._db import get_db
    with get_db() as conn:
        if fits_path is not None:
            r = conn.execute("DELETE FROM detection_cache WHERE fits_path = ?",
                             (fits_path,))
            return {"detection_entries_cleared": r.rowcount,
                    "catalog_entries_cleared": 0}
        else:
            d = conn.execute("DELETE FROM detection_cache")
            c = conn.execute("DELETE FROM catalog_cache")
            return {"detection_entries_cleared": d.rowcount,
                    "catalog_entries_cleared": c.rowcount}
