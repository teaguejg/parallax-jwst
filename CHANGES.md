# Changes

## 1.2.1

- Fixed `flux_mjy` and `mag_ab` never being persisted to the database. The
  `candidate_detections` table lacked both columns since they were added to the
  Detection dataclass. The GUI loaded candidates via `catalog.get()` and got
  None for both fields, showing `-` for every candidate. Added columns to the
  schema, migration ALTER TABLE statements, INSERT, and SELECT reconstruction.
- Verified `flux_err`, `flux_mjy_err`, and `mag_ab_err` persistence path is
  consistent (same INSERT/SELECT fix applied in the same pass).
- Fixed RuntimeError in `_on_candidate_inspected` when closing a previous
  InspectWindow whose C++ object was already deleted by `deleteLater()`.
  The `old.close()` call is now guarded with try/except RuntimeError.

## 1.2.0

- Per-source flux uncertainties propagated from the JWST i2d ERR extension
  through to Detection and Candidate. When a WHT extension is present,
  per-pixel variance is weighted (zero-weight pixels contribute nothing).
- Three new fields on Detection: `flux_err`, `flux_mjy_err`, `mag_ab_err`.
  Same three fields added to Candidate, populated from the best-SNR detection
  that has uncertainty data.
- `flux_mjy_err` is `flux_err * PIXAR_SR`. `mag_ab_err` is
  `2.5 / ln(10) * flux_mjy_err / flux_mjy`.
- Markdown report table now includes Flux err and Mag err columns.
- DetailPanel GUI shows uncertainties inline (e.g. `22.31 +/- 0.0270`).
- DB schema: `candidate_detections` and `candidates` tables gain `flux_err`,
  `flux_mjy_err`, `mag_ab_err` columns. Existing databases migrate
  automatically via ALTER TABLE.
- JSON report round-trips the new fields cleanly.
- Caveats section updated to describe ERR/WHT-based uncertainty propagation.

## 1.1.1

- Fixed stdout redirection in acquisition.py: replaced manual save/restore with
  a context manager that guarantees restoration on exceptions and
  KeyboardInterrupt. Affected `_mast_download` and `_get_expected_filenames`.
- Fixed detection cache key omitting `background_filter_size`. Changing that
  config value now correctly invalidates cached detections instead of returning
  stale results.
- Fixed JSON round-trip crash when CatalogMatch `data` field is an empty string
  instead of null or a dict. Deserialization in `report_from_dict`, `archive.py`,
  and `catalog.py` now coerces empty/null/missing values to `{}`.
- Fixed silent migration failures in `_db.py`: `init_db()` ALTER TABLE errors
  now only silence the expected "duplicate column" case and log warnings for
  anything else instead of swallowing all exceptions.
- Fixed InspectWindow cleanup race: `closeEvent` now stops matplotlib canvas
  timers, closes figures, and calls `deleteLater()` to prevent segfaults and
  QTimer warnings on rapid open/close cycles.
- Enabled `PRAGMA foreign_keys = ON` on every database connection and switched
  to WAL journal mode. Foreign key constraints on `catalog_matches`,
  `candidate_detections`, etc. are now enforced, and WAL eliminates "database
  is locked" errors from concurrent GUI reads.

## 1.1.0

- Coordinate search: `acquire()` and `reduce()` accept `ra`/`dec` kwargs for
  coordinate-based input, bypassing name resolution.
- WCS tile selection: `get_fits_per_filter` uses WCS footprint containment to
  select the correct FITS tile per filter.
- Flat downloads: files land directly in slug dir, no obs-id subfolder move.
- InspectWindow overhaul: controls split into two rows, labeled spinboxes,
  seven color palettes, Chromatic as default, custom palette memory.
- GUI toolbar accepts decimal and sexagesimal coordinates.
- Coordinate-based acquire scans existing slug dirs before querying MAST.
- Cutout2D uses partial mode with NaN fill for edge sources.
- Lupton RGB for all filter counts, per-filter alpha in ColorPickerPanel.

## 1.0.0

- Initial stable release. Full pipeline: acquire, detect, resolve, report.
- GUI with SkyPanel, DetailPanel, InspectWindow, ReportsPanel, SettingsPanel.
- SQLite persistence, detection and catalog caching.
- Cross-reference against SIMBAD, NED, and Gaia.
