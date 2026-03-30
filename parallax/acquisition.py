import logging
import os
import sys
import warnings

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import FITSFixedWarning
import astropy.units as u

from parallax.config import config
from parallax.exceptions import TargetNotFoundError
from parallax.types import _target_slug

warnings.filterwarnings("ignore", category=FITSFixedWarning)

logger = logging.getLogger(__name__)


def _resolve_name(name):
    from astroquery.simbad import Simbad
    msg = f"Cannot resolve target '{name}' - use a catalog identifier such as 'NGC 3132' or 'M92'."
    try:
        result = Simbad.query_object(name)
    except Exception:
        raise TargetNotFoundError(msg)
    if result is None or len(result) == 0:
        raise TargetNotFoundError(msg)
    main_id = str(result["main_id"][0])
    return main_id, float(result["ra"][0]), float(result["dec"][0])


def _mast_query(ra, dec, instrument, filters, max_results):
    from astroquery.mast import Observations

    coord = SkyCoord(ra, dec, unit="deg")
    # JWST instrument names in MAST include mode, e.g. "NIRCAM/IMAGE"
    inst_pattern = f"{instrument}*" if not instrument.endswith("*") else instrument
    results = Observations.query_criteria(
        coordinates=coord,
        radius=0.1 * u.deg,
        obs_collection="JWST",
        instrument_name=inst_pattern,
        calib_level=3,
        dataproduct_type="IMAGE",
    )

    # filter column is on the results table, not a query param
    if filters and len(results) > 0:
        mask = [any(f in str(row.get("filters", "")) for f in filters)
                for row in results]
        results = results[mask]

    return results[:max_results] if len(results) > 0 else results


def _mast_download(obs_rows, dest_dir, on_progress=None) -> list[str]:
    from astroquery.mast import Observations

    # mast prints progress to stdout; crashes when stdout is None (bat launcher)
    _real_stdout = sys.stdout
    _devnull = None
    try:
        try:
            _devnull = open(os.devnull, "w")
            sys.stdout = _devnull
        except Exception:
            pass
        products = Observations.get_product_list(obs_rows)
        products = Observations.filter_products(products, productType="SCIENCE")
        if len(products) == 0:
            raise RuntimeError("no i2d products found for this target")
        # keep only fully reduced mosaics, exclude per-detector _t variants
        products = products[["_i2d.fits" in str(p) for p in products["productFilename"]]]
        if len(products) == 0:
            raise RuntimeError("no i2d products found for this target")
        products = products[["_t" not in str(p) for p in products["productFilename"]]]
        if len(products) == 0:
            raise RuntimeError("no i2d products found for this target")
        os.makedirs(dest_dir, exist_ok=True)

        paths_downloaded = []
        for row in products:
            uri = str(row["dataURI"])
            filename = os.path.basename(str(row["productFilename"]))
            if on_progress:
                on_progress("downloading", filename)
            local_path = os.path.join(dest_dir, filename)
            try:
                result = Observations.download_file(uri,
                                                    local_path=local_path)
                if result[0] == "COMPLETE":
                    paths_downloaded.append(local_path)
            except Exception as e:
                logger.warning("failed to download %s: %s", filename, e)
                continue
    finally:
        sys.stdout = _real_stdout
        if _devnull is not None:
            try:
                _devnull.close()
            except Exception:
                pass

    return [str(p) for p in paths_downloaded if str(p).endswith(".fits")]


def _get_expected_filenames(obs_rows) -> set[str]:
    """Return the set of expected i2d mosaic filenames from MAST product list."""
    from astroquery.mast import Observations
    _real_stdout = sys.stdout
    try:
        _devnull = open(os.devnull, "w")
        sys.stdout = _devnull
    except Exception:
        pass
    try:
        products = Observations.get_product_list(obs_rows)
        products = Observations.filter_products(products, productType="SCIENCE")
        products = products[["_i2d.fits" in str(p)
                             for p in products["productFilename"]]]
        products = products[["_t" not in str(p)
                             for p in products["productFilename"]]]
        return {os.path.basename(str(p)) for p in products["productFilename"]}
    except Exception:
        return set()
    finally:
        sys.stdout = _real_stdout
        try:
            _devnull.close()
        except Exception:
            pass


def _validate_local_fits(paths):
    good = []
    bad = 0
    for p in paths:
        try:
            with fits.open(p) as hdul:
                hdul["SCI"].header
        except Exception:
            logger.warning("corrupt or unreadable FITS, deleting: %s", p)
            try:
                os.remove(p)
            except OSError:
                pass
            bad += 1
            continue
        good.append(p)
    if bad:
        logger.info("removed %d corrupt file(s), %d valid remaining", bad, len(good))
    return good


def _find_local_fits_covering(ra, dec, dl_path):
    from pathlib import Path
    from astropy.wcs import WCS

    jwst_root = Path(dl_path) / "mastDownload" / "JWST"
    if not jwst_root.is_dir():
        return []

    coord = SkyCoord(ra, dec, unit="deg")
    matched = []
    for p in jwst_root.rglob("*_i2d.fits"):
        if "_t" in p.name:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                with fits.open(str(p)) as hdul:
                    try:
                        sci = hdul["SCI"]
                    except KeyError:
                        sci = next(
                            (h for h in hdul if h.data is not None and h.data.ndim == 2),
                            None,
                        )
                    if sci is None:
                        continue
                    wcs = WCS(sci.header)
                    if wcs.footprint_contains(coord):
                        matched.append(str(p))
        except Exception:
            continue

    if matched:
        logger.info("found %d local file(s) covering RA=%.4f Dec=%.4f in existing slug dir",
                    len(matched), ra, dec)
    return matched


def acquire(
    target: str | None = None,
    instrument: str | None = None,
    filters: list[str] | None = None,
    max_results: int = 10,
    on_progress: callable | None = None,
    ra: float | None = None,
    dec: float | None = None,
) -> list[str]:
    """Download Level 3 FITS products from MAST matching target and instrument."""
    if target is None and (ra is None or dec is None):
        raise ValueError("target or ra/dec required")
    if target is None:
        target = ""

    if instrument is None:
        instrument = config.get("mast.instruments")[0]

    dl_path = config.get("data.download_path")

    # direct coordinate input -- skip all name resolution
    if ra is not None and dec is not None:
        slug = f"coord_{ra:.3f}_{dec:.3f}".replace("-", "m").replace(".", "p")
        slug_dir = os.path.join(dl_path, "mastDownload", "JWST", slug)

        # check existing slug dirs for files already covering these coords
        local_covering = _find_local_fits_covering(ra, dec, dl_path)
        if local_covering:
            validated = _validate_local_fits(local_covering)
            if validated:
                logger.info("reusing %d existing tile(s) covering coordinates",
                            len(validated))
                if on_progress:
                    on_progress("acquire", f"Reusing {len(validated)} cached file(s)")
                return sorted(validated)

        if on_progress:
            on_progress("acquire", "Querying MAST...")
        results = _mast_query(ra, dec, instrument, filters, max_results)
        if len(results) == 0:
            raise TargetNotFoundError(
                f"no JWST Level 3 data found at RA={ra}, Dec={dec}"
            )

        if os.path.isdir(slug_dir):
            from pathlib import Path
            local = [str(p) for p in Path(slug_dir).rglob("*_i2d.fits")
                     if "_t" not in p.name]
            local = _validate_local_fits(local)
            if local:
                expected = _get_expected_filenames(results)
                local_names = {os.path.basename(p) for p in local}
                if expected and local_names >= expected:
                    logger.info("cache complete (%d file(s)), skipping download",
                                len(local))
                    return sorted(local)
                elif expected:
                    logger.info("cache incomplete (%d/%d), downloading missing files",
                                len(local), len(expected))
                else:
                    logger.info("could not verify cache completeness, re-downloading")

        paths = _mast_download(results, slug_dir, on_progress=on_progress)
        if not paths:
            raise RuntimeError("download completed but no FITS files found")
        return paths

    slug = _target_slug(target)
    canonical_slug = slug

    if "/" in target or target.strip().isdigit():
        slug_dir = os.path.join(dl_path, "mastDownload", "JWST", slug)

        if on_progress:
            on_progress("acquire", "Querying MAST...")
        from astroquery.mast import Observations
        results = Observations.query_criteria(
            obs_id=target,
            calib_level=[3],
            dataproduct_type="IMAGE",
        )
        if len(results) == 0:
            raise ValueError(f"no products found for obs_id {target}")

        if os.path.isdir(slug_dir):
            from pathlib import Path
            local = [str(p) for p in Path(slug_dir).rglob("*_i2d.fits")
                     if "_t" not in p.name]
            local = _validate_local_fits(local)
            if local:
                expected = _get_expected_filenames(results[:max_results])
                local_names = {os.path.basename(p) for p in local}
                if expected and local_names >= expected:
                    logger.info("cache complete (%d file(s)), skipping download",
                                len(local))
                    return sorted(local)
                elif expected:
                    logger.info("cache incomplete (%d/%d), downloading missing files",
                                len(local), len(expected))
                else:
                    logger.info("could not verify cache completeness, re-downloading")

        paths = _mast_download(results[:max_results], slug_dir,
                               on_progress=on_progress)
    else:
        parts = target.strip().split()
        if len(parts) == 2:
            try:
                ra, dec = float(parts[0]), float(parts[1])
            except ValueError:
                main_id, ra, dec = _resolve_name(target)
                canonical_slug = _target_slug(main_id)
        else:
            main_id, ra, dec = _resolve_name(target)
            canonical_slug = _target_slug(main_id)

        slug_dir = os.path.join(dl_path, "mastDownload", "JWST", canonical_slug)

        if on_progress:
            on_progress("acquire", "Querying MAST...")
        results = _mast_query(ra, dec, instrument, filters, max_results)
        if len(results) == 0:
            raise TargetNotFoundError(
                f"no JWST Level 3 data found for '{target}'"
            )

        if os.path.isdir(slug_dir):
            from pathlib import Path
            local = [str(p) for p in Path(slug_dir).rglob("*_i2d.fits")
                     if "_t" not in p.name]
            local = _validate_local_fits(local)
            if local:
                expected = _get_expected_filenames(results)
                local_names = {os.path.basename(p) for p in local}
                if expected and local_names >= expected:
                    logger.info("cache complete (%d file(s)), skipping download",
                                len(local))
                    return sorted(local)
                elif expected:
                    logger.info("cache incomplete (%d/%d), downloading missing files",
                                len(local), len(expected))
                else:
                    logger.info("could not verify cache completeness, re-downloading")

        paths = _mast_download(results, slug_dir, on_progress=on_progress)

    if not paths:
        raise RuntimeError("download completed but no FITS files found")
    return paths
