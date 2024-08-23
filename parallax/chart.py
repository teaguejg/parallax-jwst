import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from parallax.types import Candidate, Report, report_from_dict

logger = logging.getLogger(__name__)

_COLORS = {
    "unverified": "#c0392b",
    "known": "#a8d8ea",
}

_MARKERS = ['o', 's', '^', 'D', 'v']


def _load_report(report) -> Report:
    if isinstance(report, Report):
        return report
    with open(report) as f:
        return report_from_dict(json.load(f))


def plot(
    report: Report | str,
    show_known: bool = False,
    output_path: str | None = None,
) -> None:
    """Spatial scatter plot of candidates from a single report."""
    rpt = _load_report(report)

    cands = rpt.candidates
    if not show_known:
        cands = [c for c in cands if c.classification != "known"]
    if not cands:
        raise ValueError("no candidates to plot after filtering")

    fig, ax = plt.subplots(figsize=(8, 8))
    for cls, color in _COLORS.items():
        subset = [c for c in cands if c.classification == cls]
        if not subset:
            continue
        ras = [c.ra for c in subset]
        decs = [c.dec for c in subset]
        snrs = np.array([c.snr for c in subset])
        sizes = np.clip(snrs * 10, 20, 200)
        alphas = np.array([0.3 + 0.6 * c.confidence for c in subset])
        ax.scatter(ras, decs, s=sizes, c=color, label=cls, alpha=alphas, edgecolors="k", linewidth=0.5)

    ax.invert_xaxis()
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"Survey: {rpt.target} / {rpt.instrument}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def overlay(
    reports: list[Report | str],
    output_path: str | None = None,
) -> None:
    """Overlay candidates from multiple reports on a shared coordinate space."""
    if not reports:
        raise ValueError("reports list is empty")

    loaded = [_load_report(r) for r in reports]

    if len(loaded) > 1:
        ras = [[c.ra for c in r.candidates if c.ra == c.ra] for r in loaded]
        if all(ras):
            mins = [min(r) if r else 0 for r in ras]
            maxs = [max(r) if r else 0 for r in ras]
            if max(mins) > min(maxs):
                logger.warning("reports may cover non-overlapping sky areas")

    fig, ax = plt.subplots(figsize=(8, 8))

    for i, rpt in enumerate(loaded):
        marker = _MARKERS[i % len(_MARKERS)]
        for cls, color in _COLORS.items():
            subset = [c for c in rpt.candidates if c.classification == cls]
            if not subset:
                continue
            ras = [c.ra for c in subset]
            decs = [c.dec for c in subset]
            snrs = np.array([c.snr for c in subset])
            sizes = np.clip(snrs * 10, 20, 200)
            alphas = np.array([0.3 + 0.6 * c.confidence for c in subset])
            lbl = f"{rpt.target} ({cls})" if i == 0 or cls == "unverified" else None
            ax.scatter(ras, decs, s=sizes, c=color, marker=marker,
                      label=lbl, alpha=alphas, edgecolors="k", linewidth=0.5)

    ax.invert_xaxis()
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    # TODO: auto-generate title from report targets
    ax.set_title("Multi-report overlay")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


def field(
    ra: float,
    dec: float,
    radius_deg: float,
    candidates: list[Candidate] | None = None,
    output_path: str | None = None,
) -> None:
    """Plot a sky field with optional candidate overlay."""
    fig, ax = plt.subplots(figsize=(8, 8))
    _skyview_ok = False

    try:
        from astroquery.skyview import SkyView
        import astropy.units as u
        imgs = SkyView.get_images(
            position=f"{ra} {dec}",
            coordinates="J2000",
            survey=["DSS"],
            radius=radius_deg * u.deg,
        )
        if imgs and len(imgs) > 0:
            ax.imshow(imgs[0][0].data, cmap="gray", origin="lower",
                     extent=[ra + radius_deg, ra - radius_deg,
                             dec - radius_deg, dec + radius_deg],
                     aspect="auto")
            _skyview_ok = True
    except Exception as e:
        logger.warning("SkyView unavailable, using plain grid: %s", e)

    if not _skyview_ok:
        ax.set_xlim(ra + radius_deg, ra - radius_deg)
        ax.set_ylim(dec - radius_deg, dec + radius_deg)

    if candidates:
        for cls, color in _COLORS.items():
            subset = [c for c in candidates if c.classification == cls]
            if not subset:
                continue
            ax.scatter([c.ra for c in subset], [c.dec for c in subset],
                      c=color, label=cls, edgecolors="k", linewidth=0.5, zorder=5)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"Field: {ra:.4f}, {dec:.4f}")
    ax.grid(True, alpha=0.3)
    if candidates:
        ax.legend()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
