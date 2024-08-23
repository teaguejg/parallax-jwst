import os
import tempfile
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


def make_fits(shape=(200, 200), n_sources=3, noise=0.1):
    rng = np.random.default_rng(42)
    data = rng.normal(0, noise, shape).astype(np.float32)

    # plant point sources
    for _ in range(n_sources):
        x = rng.integers(20, shape[1] - 20)
        y = rng.integers(20, shape[0] - 20)
        data[y-2:y+3, x-2:x+3] += rng.uniform(5, 20) * noise

    w = WCS(naxis=2)
    w.wcs.crpix = [shape[1] // 2, shape[0] // 2]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    w.wcs.crval = [83.8221, -5.3911]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    hdr = w.to_header()
    hdr["INSTRUME"] = "NIRCAM"
    hdr["FILTER"] = "F200W"
    hdr["OBSERVTN"] = "TEST001"

    # mimic JWST i2d layout: empty primary + named SCI extension
    primary = fits.PrimaryHDU()
    sci = fits.ImageHDU(data, header=hdr, name="SCI")
    return fits.HDUList([primary, sci])


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield d


@pytest.fixture
def tmp_db(tmp_dir):
    """Set up an isolated config + DB for testing."""
    from parallax.config import config
    from parallax._db import init_db

    db_path = os.path.join(tmp_dir, "test.db")
    config._data = {
        "mast": {"instruments": ["NIRCAM", "MIRI"], "calib_level": 3, "product_type": "IMAGE"},
        "detection": {"snr_threshold": 3.0, "min_pixels": 25, "cutout_padding": 20,
                      "kernel_fwhm": 2.0, "cutout_size": 60},
        "resolver": {"search_radius_arcsec": 2.0, "catalogs": ["SIMBAD", "NED", "GAIA"],
                      "timeout_seconds": 30},
        "report": {"output_format": "both", "include_known": False},
        "cache": {"detection_enabled": True, "catalog_enabled": True,
                  "catalog_ttl_days": 30, "candidate_match_radius_arcsec": 2.0},
        "data": {
            "download_path": os.path.join(tmp_dir, "downloads"),
            "processed_path": os.path.join(tmp_dir, "processed"),
            "reports_path": os.path.join(tmp_dir, "reports"),
            "archive_path": os.path.join(tmp_dir, "archive"),
            "db_path": db_path,
        },
    }
    config._loaded = True

    for sub in ["downloads", "processed", "reports", "archive", "archive/cutouts"]:
        os.makedirs(os.path.join(tmp_dir, sub), exist_ok=True)

    init_db()
    yield tmp_dir
