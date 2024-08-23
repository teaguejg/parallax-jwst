"""
Parallax walkthrough -- Globular Cluster M92

Runs the full pipeline step by step with printed commentary at each stage.
Results are saved to the data/ directory as configured in config.yaml.

M92 (NGC 6341) is a globular cluster in Hercules observed by JWST NIRCam
as part of the Early Release Science program for resolved stellar populations.
It has confirmed Level 3 public data in MAST and is dense enough that
detection will find many sources with a mix of classifications.

Run from the parallax project root:
    python examples/explore_m92.py
"""

import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from astropy.io import fits as astropy_fits
from astropy.wcs import FITSFixedWarning
warnings.filterwarnings("ignore", category=FITSFixedWarning)

import parallax as par
from parallax.config import config
from parallax.exceptions import TargetNotFoundError
from parallax.logger import RunLogger
from parallax.survey import _merge_detections

TARGET = "M92"
INSTRUMENT = "NIRCAM"

log_path = config.get("log.path", "data/parallax.log")
max_runs = config.get("log.max_runs", 10)
rlog = RunLogger(log_path, max_runs)
rlog.start(par.__version__, TARGET)

def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


# Step 1: Acquire
section("Step 1 of 7: Download FITS data from MAST")

print(f"  Target:     {TARGET}  (NGC 6341, globular cluster in Hercules)")
print(f"  Instrument: {INSTRUMENT}")
print(f"  Calib level: 3 (fully reduced, science-ready)")
print()

try:
    fits_paths = par.survey.acquire(TARGET, instrument=INSTRUMENT)
except ConnectionError:
    rlog.step("step 1  acquire", "[fail] Could not reach MAST")
    rlog.end()
    print("  [fail] Could not reach MAST. Check your internet connection and try again.")
    sys.exit(1)
except (ValueError, TargetNotFoundError) as e:
    rlog.step("step 1  acquire", f"[fail] {e}")
    rlog.end()
    print(f"  [fail] {e}")
    sys.exit(1)

total_sz = sum(os.path.getsize(p) for p in fits_paths)
print(f"  [ok] {len(fits_paths)} i2d file(s), total size: {total_sz / 1e6:.1f} MB")
rlog.step("step 1  acquire", f"{len(fits_paths)} files")


# Step 2: Cache status
section("Step 2 of 7: Cache status")

status = par.survey.cache_status()
det_entries = status.get("detection", [])
cat_entries = status.get("catalog", [])

if not det_entries and not cat_entries:
    print("  No cached results yet. This run will query live catalogs.")
    print("  Run the script again after completion to see caching in action.")
else:
    print(f"  Detection cache: {len(det_entries)} file(s) cached")
    for e in det_entries:
        name = os.path.basename(e["fits_path"])
        print(f"    {name} -- {e['detections_count']} sources, cached {e['cached_at'][:10]}")

    print(f"  Catalog cache:  {len(cat_entries)} field/catalog entries")
    catalogs_cached = set(e["catalog"] for e in cat_entries)
    if catalogs_cached:
        print(f"    Catalogs: {', '.join(sorted(catalogs_cached))}")
        print(f"    This run's catalog queries will be served from cache.")

rlog.step("step 2  cache status",
          f"{len(det_entries)} detection entries, {len(cat_entries)} catalog entries")
print()


# Step 3: Detect (per-filter)
section("Step 3 of 7: Source detection (per filter)")

print("  This uses background subtraction and a Gaussian convolution kernel")
print("  before thresholding. Sources below 3x background noise are ignored.")
print()

i2d_files = [p for p in fits_paths if "i2d" in os.path.basename(p)]
if not i2d_files:
    i2d_files = fits_paths[:1]

all_detections = []
fits_inputs = []

t0 = time.monotonic()
for fp in i2d_files:
    with astropy_fits.open(fp) as hdul:
        hdr = hdul[0].header
        filt = hdr.get("FILTER", hdr.get("FILTER1", "UNKNOWN"))
    fits_inputs.append((fp, filt))
    dets = par.survey.detect(fp, filter_name=filt)
    print(f"  {filt}: {len(dets)} sources in {os.path.basename(fp)}")
    all_detections.extend(dets)

# merge across filters
merged = _merge_detections(all_detections)
detect_elapsed = time.monotonic() - t0
detect_was_cached = detect_elapsed < 1.0

print()
print(f"  Raw detections (all filters): {len(all_detections)}")
print(f"  After merge by position:      {len(merged)}")

if not merged:
    rlog.step("step 3  detect", "0 sources", cache_hit=detect_was_cached, elapsed=detect_elapsed)
    print("  No sources detected above the SNR threshold.")
    rlog.end()
    sys.exit(0)

rlog.step("step 3  detect", f"{len(merged)} merged sources", cache_hit=detect_was_cached, elapsed=detect_elapsed)

# show a sample
sample = merged[:5]
print()
print(f"  First {len(sample)} merged detections:")
print(f"  {'RA':>12}  {'Dec':>12}  {'SNR':>6}  {'Flux':>10}  {'Filters':>12}")
print(f"  {'-'*12}  {'-'*12}  {'-'*6}  {'-'*10}  {'-'*12}")
for d in sample:
    ra = f"{d['ra']:.4f}" if d['ra'] == d['ra'] else "NaN"
    dec = f"{d['dec']:.4f}" if d['dec'] == d['dec'] else "NaN"
    filts = ",".join(dd.get("filter", "?") for dd in d.get("detections", []))
    print(f"  {ra:>12}  {dec:>12}  {d['snr']:>6.1f}  {d['flux']:>10.1f}  {filts:>12}")

if len(merged) > 5:
    print(f"  ... and {len(merged) - 5} more")


# Step 4: Resolve
section("Step 4 of 7: Catalog cross-reference")

print("  Querying SIMBAD, NED, and Gaia for each detected source.")
print("  Search radius: 2.0 arcseconds (configurable in config.yaml)")
print()

t0 = time.monotonic()
try:
    candidates, gaia_flag = par.survey.resolve(merged)
except ConnectionError:
    rlog.step("step 4  resolve", "[fail] all catalog queries failed")
    rlog.end()
    print("  [fail] All three catalogs were unreachable. Check your connection.")
    sys.exit(1)
resolve_elapsed = time.monotonic() - t0

unverified = [c for c in candidates if c.classification == "unverified"]
known = [c for c in candidates if c.classification == "known"]

rlog.step("step 4  resolve",
          f"{len(unverified)} unverified / {len(known)} known",
          elapsed=resolve_elapsed)

print(f"  [ok] Classification complete")
print()
print(f"  Unverified:  {len(unverified):>4}  (no catalog match)")
print(f"  Known:       {len(known):>4}  (matched in catalog)")


# Step 5: Report
section("Step 5 of 7: Build and save report")

print("  Generating report (JSON + Markdown) and saving candidates to the database.")
print()

report = par.survey.report(candidates, TARGET, fits_inputs, len(merged), gaia_failed=gaia_flag)

rlog.step("step 5  report", report.id)

print(f"  [ok] Report ID: {report.id}")
if report.json_path:
    print(f"       JSON:      {report.json_path}")
if report.md_path:
    print(f"       Markdown:  {report.md_path}")
print()
print(f"  Summary:")
print(f"    Sources detected:  {report.n_sources_detected}")
print(f"    Catalog matched:   {report.n_catalog_matched}")
print(f"    Unverified:        {report.n_unverified}")
print(f"    Filters:           {' '.join(report.filters)}")

if report.md_path:
    print()
    print(f"  Open {report.md_path} in any text editor or Markdown viewer")
    print("  for a formatted summary.")


# Step 6: View
section("Step 6 of 7: Visual inspection")

report_dir = os.path.dirname(report.md_path) if report.md_path else config.get("data.reports_path")

if not unverified:
    print("  No unverified candidates to inspect.")
    rlog.step("step 6  view", "no unverified candidates")
else:
    session = par.view.open(report)

    # try candidates in order until we find one with a valid cutout
    cutout = None
    candidate = None
    to_try = unverified[:50]
    for c in to_try:
        print(f"  Trying candidate {c.id}...")
        try:
            cv = par.view.examine(c, session)
            cutout_png = os.path.join(report_dir, f"{report.id}_cutout.png")
            par.view.show(cv, output_path=cutout_png)
            cutout = cv
            candidate = c
            break
        except (ValueError, FileNotFoundError, Exception) as e:
            print(f"    skipped ({e})")

    if cutout is not None:
        print()
        print(f"  Showing candidate {candidate.id}")
        print(f"  RA={candidate.ra:.4f}  Dec={candidate.dec:.4f}  SNR={candidate.snr:.1f}")
        print(f"  [ok] Cutout saved to {cutout_png}")
        rlog.step("step 6  view", f"{candidate.id} cutout ok")
    else:
        print(f"  [fail] None of the first {len(to_try)} candidates produced a valid cutout.")
        print("  This can happen when candidates fall on detector gaps or image edges.")
        rlog.step("step 6  view", f"no valid cutout in first {len(to_try)}")

    print()
    chart_png = os.path.join(report_dir, f"{report.id}_chart.png")
    par.chart.plot(report, output_path=chart_png)
    print(f"  [ok] Sky chart saved to {chart_png}")
    if cutout is not None:
        print()
        print("  Open both PNG files to inspect the candidate and its position")
        print("  relative to other sources in the field.")


# Step 7: Tag and annotate
section("Step 7 of 7: Tag and annotate")

if not unverified:
    print("  No unverified candidates to tag.")
    rlog.step("step 7  tag", "no unverified candidates")
else:
    candidate = unverified[0]
    print(f"  Tagging {candidate.id} for follow-up.")
    print()

    par.archive.tag(candidate.id, ["followup", "m92-run-1"])
    par.archive.annotate(candidate.id, "first unverified candidate from M92 NIRCam survey")

    updated = par.catalog.get(candidate.id)
    print(f"  [ok] Tags:  {updated.tags}")
    print(f"       Notes: {updated.notes}")
    print()

    print("  Candidate history (all changes recorded automatically):")
    h = par.catalog.history(candidate.id)
    for entry in h:
        print(f"    {entry['timestamp'][:19]}  {entry['field']}  ->  {entry['new_value']}")
    rlog.step("step 7  tag", f"{candidate.id} tagged")


# Done
section("Done")

rlog.end()

print(f"  Report ID:  {report.id}")
print(f"  Candidates: {len(candidates)} total, {report.n_unverified} unverified")
print(f"  Run log:    {log_path}")
print()
print("  Output files:")
if report.json_path:
    print(f"    {report.json_path}")
if report.md_path:
    print(f"    {report.md_path}")
if 'cutout_png' in dir():
    print(f"    {cutout_png}")
if 'chart_png' in dir():
    print(f"    {chart_png}")
print()
print("  Everything is stored in data/. To pick up where you left off:")
print()
print(f"    import parallax as par")
print(f"    report = par.archive.get_report('{report.id}')")
print(f"    session = par.view.open(report)")
print()
print("  To export your candidates:")
print()
print(f"    par.archive.export('{report.id}', format='csv')")
print()
print("  To set up a watch for new M92 observations:")
print()
print("    par.monitor.watch({")
print('        "name": "M92 watch",')
print('        "instruments": ["NIRCAM"],')
print("        \"ra\": 259.2808,")
print("        \"dec\": 43.1358,")
print("        \"radius_deg\": 0.1")
print("    })")
print("    par.monitor.start()")
print()
