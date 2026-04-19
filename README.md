# Parallax

Parallax is a Python library for exploring publicly available JWST data from
the MAST archive. It finds astronomical sources that existing catalogs have
not yet characterized, cross-references them against SIMBAD, NED, and Gaia,
and presents the results in terms an educated non-specialist can act on.

## Installation

Requires Python 3.12 or later.

```
git clone https://github.com/teaguejg/parallax-jwst
cd parallax-jwst
pip install -e .
```

This installs all dependencies. PyQt6 is included for the desktop GUI.

## Quick start

```python
import parallax as par

# run the full pipeline on M92
report = par.survey.reduce("M92", instrument="NIRCAM")

print(f"Detected: {report.n_sources_detected}")
print(f"Unverified: {report.n_unverified}")

# launch the GUI
from parallax.gui import launch
launch()
```

`reduce()` downloads JWST images from MAST, runs source detection, queries
three catalogs, and writes JSON and markdown reports to `data/reports/`.

## Validated targets

These targets have been tested end-to-end:

- **M92 (NGC 6341)** - globular cluster. ~23,800 sources detected, ~18,700
  unverified candidates. Runs in under two minutes.
- **NGC 3132** - Southern Ring Nebula. Use "NGC 3132" as the target string;
  "Southern Ring Nebula" does not resolve on MAST.
- **Orion Bar** - star-forming region. MAST has no Level 3 mosaic products
  for this target, only detector-level i2d files.

## Configuration

All settings live in `config.yaml` in the project root. Key knobs:

- `detection.snr_threshold` (default 3.0) - minimum signal-to-noise for a
  detection. Lower finds fainter sources but more noise.
- `detection.min_pixels` (default 5) - minimum connected pixel area for a
  source to be reported. Raise to suppress noise detections.
- `detection.background_box_size` (default 50) - background estimation tile
  size. Smaller values (20-25) work better near bright extended emission.
- `resolver.search_radius_arcsec` (default 2.0) - catalog cross-match radius.
  Increase if WCS alignment is imprecise.

Environment variables override config values using the prefix `PARALLAX_` with
underscores between segments (e.g. `PARALLAX_DETECTION_SNR_THRESHOLD=5.0`).

## Requirements

- astropy >= 5.3
- astroquery >= 0.4.8
- photutils >= 1.9
- matplotlib >= 3.7
- numpy >= 1.24
- PyYAML >= 6.0
- scipy >= 1.11
- PyQt6 >= 6.4
