import builtins
import json
import logging
import os
import tempfile
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS, FITSFixedWarning
from astropy.visualization import AsinhStretch, LogStretch, LinearStretch, ImageNormalize, ZScaleInterval
import astropy.units as u

warnings.filterwarnings("ignore", category=FITSFixedWarning)

from parallax.config import config
from parallax.types import (
    Candidate, Report, CutoutView, ViewSession,
    report_from_dict,
)

logger = logging.getLogger(__name__)
_builtin_open = builtins.open

try:
    from IPython import get_ipython
    _in_notebook = get_ipython() is not None
except ImportError:
    _in_notebook = False

_STRETCHES = {
    "asinh": AsinhStretch,
    "log": LogStretch,
    "linear": LinearStretch,
}


def _find_sci_hdu(hdul):
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


def _resolve_fits_path(report):
    from parallax._db import get_db
    with get_db() as conn:
        row = conn.execute(
            "SELECT fits_path FROM report_inputs WHERE report_id = ? ORDER BY id LIMIT 1",
            (report.id,)
        ).fetchone()
        if row and row["fits_path"] and os.path.isfile(row["fits_path"]):
            return row["fits_path"]
        # fall back to old reports.fits_path column
        rrow = conn.execute(
            "SELECT fits_path FROM reports WHERE id = ?", (report.id,)
        ).fetchone()
        if rrow and rrow["fits_path"] and os.path.isfile(rrow["fits_path"]):
            return rrow["fits_path"]
    return None


def open(report: Report | str) -> ViewSession:
    """Load a report and its FITS data for examination."""
    if isinstance(report, str):
        if not os.path.isfile(report):
            raise FileNotFoundError(report)
        with _builtin_open(report) as f:
            d = json.load(f)
        report = report_from_dict(d)

    fits_path = _resolve_fits_path(report)
    if fits_path is None:
        raise FileNotFoundError(f"no FITS file found for report {report.id}")

    hdul = fits.open(fits_path)
    return ViewSession(report, hdul)



def examine(
    candidate: Candidate | str,
    session: ViewSession | None = None,
    output_path: str | None = None,
) -> CutoutView:
    """Extract a cutout around a candidate source."""
    if isinstance(candidate, str):
        if session is not None:
            cand = next((c for c in session.candidates if c.id == candidate), None)
            if cand is None:
                raise ValueError(f"candidate {candidate} not in session")
            candidate = cand
        else:
            from parallax import catalog, archive
            candidate = catalog.get(candidate)
            if candidate is None:
                raise KeyError("candidate not found")
            fits_path = archive.get_fits(candidate.id)
            hdul = fits.open(fits_path)
            try:
                sci = _find_sci_hdu(hdul)
                wcs = WCS(sci.header) if sci else None
                data = sci.data.astype(np.float64) if sci else None
                size = config.get("detection.cutout_size", 60)
                coord = SkyCoord(candidate.ra, candidate.dec, unit="deg")
                cutout = Cutout2D(data, coord, size * u.pixel, wcs=wcs)
                border = _border_median(cutout.data, 3)
            finally:
                hdul.close()
            result = cutout.data - border
            if np.all(np.isnan(result)):
                raise ValueError("cutout data is empty or contains no valid pixels")
            return CutoutView(candidate, result, cutout.wcs,
                              fits_path, cutout.data.shape)

    if session is not None:
        sci = _find_sci_hdu(session.fits)
        data = sci.data.astype(np.float64)
        wcs = WCS(sci.header)
        try:
            fits_path = _resolve_fits_path(session.report) or "unknown"
        except Exception:
            fits_path = "unknown"
    elif isinstance(candidate, Candidate):
        from parallax import archive
        fits_path = archive.get_fits(candidate.id)
        hdul = fits.open(fits_path)
        try:
            sci = _find_sci_hdu(hdul)
            data = sci.data.astype(np.float64)
            wcs = WCS(sci.header)
        finally:
            hdul.close()
    else:
        raise FileNotFoundError("no FITS available")

    size = config.get("detection.cutout_size", 60)
    coord = SkyCoord(candidate.ra, candidate.dec, unit="deg")
    cutout = Cutout2D(data, coord, size * u.pixel, wcs=wcs)
    border = _border_median(cutout.data, 3)
    result = cutout.data - border

    if np.all(np.isnan(result)):
        raise ValueError("cutout data is empty or contains no valid pixels")

    return CutoutView(candidate, result, cutout.wcs,
                      fits_path, cutout.data.shape)


def _border_median(data, width):
    mask = np.ones_like(data, dtype=bool)
    if data.shape[0] > 2*width and data.shape[1] > 2*width:
        mask[width:-width, width:-width] = False
    vals = data[mask]
    if np.all(np.isnan(vals)):
        return 0.0
    return float(np.nanmedian(vals))


def show(
    cutout_view: CutoutView,
    stretch: str = "asinh",
    colormap: str = "gray",
    output_path: str | None = None,
) -> None:
    """Display or save a cutout image."""
    if stretch not in _STRETCHES:
        raise ValueError(f"unknown stretch: {stretch}")

    d = cutout_view.data
    if d.size == 0 or np.all(np.isnan(d)):
        raise ValueError("cutout data is empty or contains no valid pixels")

    stretch_obj = _STRETCHES[stretch]()
    norm = ImageNormalize(d, interval=ZScaleInterval(),
                          stretch=stretch_obj)

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": cutout_view.wcs})
    ax.imshow(cutout_view.data, origin="lower", cmap=colormap, norm=norm)
    c = cutout_view.candidate
    ax.set_title(f"{c.id}  |  RA {c.ra:.4f}  Dec {c.dec:.4f}  SNR {c.snr:.1f}")
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    elif _in_notebook:
        from IPython.display import display
        display(fig)
        plt.close(fig)
    else:
        tmp = tempfile.mkstemp(suffix=".png")
        os.close(tmp[0])
        fig.savefig(tmp[1], dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(tmp[1])


def compare(
    candidates: list[Candidate | str],
    session: ViewSession | None = None,
    stretch: str = "asinh",
    output_path: str | None = None,
) -> None:
    """Display cutouts for multiple candidates side by side."""
    if not candidates:
        raise ValueError("candidates list is empty")

    cutouts = [examine(c, session) for c in candidates]
    n = len(cutouts)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    if stretch not in _STRETCHES:
        raise ValueError(f"unknown stretch: {stretch}")
    stretch_obj = _STRETCHES[stretch]()

    for ax, cv in zip(axes, cutouts):
        norm = ImageNormalize(cv.data, interval=ZScaleInterval(),
                              stretch=stretch_obj)
        ax.imshow(cv.data, origin="lower", cmap="gray", norm=norm)
        ax.set_title(f"{cv.candidate.id}\n{cv.candidate.classification}", fontsize=9)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    elif _in_notebook:
        from IPython.display import display
        display(fig)
        plt.close(fig)
    else:
        # TODO: handle non-notebook display better
        tmp = tempfile.mkstemp(suffix=".png")
        os.close(tmp[0])
        fig.savefig(tmp[1], dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(tmp[1])
