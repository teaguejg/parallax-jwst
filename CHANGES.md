# Changes

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
