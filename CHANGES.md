# Changes

## 1.4.0

- Target name normalization now handles lowercase catalog prefixes (e.g.
  "ngc 3132" resolves correctly in addition to "NGC 3132"). Whitespace is
  stripped before matching.
- SkyPanel replaced scatter plot with a FITS composite as the default sky view.
  The panel loads FITS files for the active report, reprojects them onto a shared
  WCS frame using `reproject`, and displays the coadded image on a plain axes
  with RA/Dec tick labels derived from the WCS (x-axis inverted for RA).
- Candidate markers overlay the sky image using WCS coordinate-to-pixel
  conversion. Classification coloring, bookmarked/viewed overlays, selection
  markers, and hit-testing all carry over from the scatter implementation.
- Falls back to the scatter plot when FITS data is unavailable or WCS fails.
- Composite is built off the main thread (SkyCompositeWorker) with a
  "Loading sky view..." label shown while processing.
- The "Field" / "Scatter" / "Both" mode cycling removed; single view replaces it.
- New dependency: `reproject>=0.13`.
- 13 new tests for sky view logic (fallback, marker positions, coloring, signals).
- `get_fits_for_report` now returns `dict[str, list[str]]` (all on-disk paths
  per filter). SkyCompositeWorker builds a multi-detector mosaic from the filter
  with the most files using `reproject.mosaicking.find_optimal_celestial_wcs` and
  `reproject_and_coadd`. Falls back to a single tile if mosaicking fails.
  Tiles are downsampled 4x via strided slicing in `_load_tiles` with matching
  WCS adjustments (CRPIX scaled, CD matrix or CDELT scaled by 4) to reduce
  memory and compute.
- SessionLogHandler now uses `self.format(record)` instead of
  `record.getMessage()`, so exception tracebacks appear in the LogBar.
- GUI launch adds a `logging.FileHandler` at `data/parallax.log` (truncated
  each launch, DEBUG level) so all log output including tracebacks is captured
  to disk.
- Fixed LogBar message duplication: progress events were reaching the log bar
  through both the logging handler signal and a direct `append()` call in
  `_on_progress`. Removed the redundant direct call.
- Fixed selection marker crash: `_draw_selection_marker` used `ax.scatter`
  whose `PathCollection.remove()` raises `NotImplementedError` in current
  matplotlib. Replaced with `ax.plot` returning a `Line2D`, which removes
  cleanly.
- Removed sky composite downsampling and its broken WCS adjustment. The
  `_downsample` helper patched `crpix`/`cdelt` without updating the PC matrix,
  producing incorrect marker positions on JWST images. Full-resolution arrays
  are now passed through to the display.
- Sky view uses plain axes with WCS-derived RA/Dec tick labels instead of
  WCSAxes projection, avoiding conflicts between WCSAxes and manual axis limit
  control. Zoom, pan, and `set_xlim`/`set_ylim` operate in pixel coordinates.
- Initial draw shows the full image extent from `imshow` defaults.
  `invert_xaxis()` runs after marker overlay but before limits are captured,
  so Reset restores the full inverted view.
- Replaced `tight_layout()` with fixed `subplots_adjust` margins to eliminate
  layout warnings.
- Right-click drag panning on the sky view. Press captures display-pixel
  coordinates and current axis limits; move computes the pixel delta and
  converts to data units via `transData.inverted()` to avoid jitter from
  shifting data coordinates; release ends the pan.
- Fixed stale `_selected_marker` reference after `fig.clear()`:
  `_draw_wcs_view` now clears the marker before clearing the figure,
  preventing `.remove()` calls on a dead artist.
- Fixed overlay refreshes (bookmark toggle, viewed tag) resetting the zoom
  level. `_draw_wcs_view` now captures axis limits before redraw and
  restores them after when the view has already been drawn once.
- Fixed WCS downsampling in `_load_tiles`: NIRCam i2d files use a CD matrix
  for pixel scale, not CDELT. The code now checks `wcs.wcs.has_cd()` and
  scales the CD matrix by 4 when present, falling back to CDELT scaling
  otherwise. CRPIX adjustment corrected to `(old - 1) / 4 + 1` per axis.
  This fixes incorrect `all_world2pix` marker placement on CD-matrix tiles.
- Zoom level indicator in WCS sky view: a text label in the lower-left corner
  shows the current zoom ratio (e.g. "2.3x") when zoomed in or out. Hidden
  at 1.0x. Updates on scroll, zoom buttons, pan release, and reset. Not shown
  in scatter fallback mode.

## 1.3.1

- `local_rms` field added to Detection dataclass, persisted to DB and JSON.
  Per-filter background RMS is now carried from detect() through merge/resolve
  to the Detection object.
- Per-filter background variance warning in markdown reports: when local_rms
  varies more than 3x across a candidate's filters, a warning line is emitted
  after the hints block.
- Composite save sidecar: InspectWindow now writes a `_composite.json`
  alongside the PNG with filter-to-hex-color mapping, per-filter alpha, stretch,
  Q, palette name, and parallax version for composite reproduction.
- Enhanced inspection summary: `_summary.md` now includes confidence score and
  quality label, auto-tags, hints, and notes between the Candidate and
  Detections sections.
- Fixed markdown hints format: `**{id}** **Hints:**` changed to
  `**{id}** - Hints:` (single bold for ID, plain text for rest).

## 1.3.0

- Sub-classification hints: short human-readable strings attached to unverified
  candidates describing observable signal properties (morphology, narrowband
  excess, single-filter). Hints appear in the markdown report, DetailPanel,
  and InspectWindow.
- Morphology properties (elongation, ellipticity, semimajor_sigma) extracted
  from SourceCatalog and carried through detection and merge pipelines.
- Hint types: `extended in {filter}`, `point source in {filter}`,
  `narrowband-only detection`, `single-filter detection`,
  `F187N excess vs continuum`, `F162M excess vs continuum`.
- New `hints` field on Candidate (list of strings), persisted to DB and JSON.
- DetailPanel shows hints below auto-tags when present.
- InspectWindow shows hints as a muted label below the filter strip.
- Fixed Repository and Issues URLs in pyproject.toml to point to the public
  repo.

## 1.2.3

- SNR now computed from local RMS at the source centroid rather than global
  background RMS. Falls back to global RMS when the map is unavailable.
- Confidence scorer now reads `resolver.search_radius_arcsec` from config
  instead of a hardcoded 2.0 arcsec default.

## 1.2.2

- Strip cutouts in InspectWindow are clickable: clicking a filter panel
  toggles that filter on/off and rebuilds the composite.
- "Mark viewed" / "Viewed" toggle button added to DetailPanel.
- Viewed candidates shown as a green scatter layer in sky panel.
- `refresh_bookmarks` renamed to `refresh_overlays`.

## 1.2.1

- Fixed `flux_mjy` and `mag_ab` not persisting to the database. Added missing
  columns to `candidate_detections`, migration ALTER TABLE statements, and
  corrected INSERT/SELECT.
- Fixed RuntimeError when closing an InspectWindow whose Qt object was already
  deleted. `old.close()` is now guarded with try/except RuntimeError.

## 1.2.0

- Flux uncertainties propagated from the JWST i2d ERR extension through to
  Detection and Candidate. WHT extension used for per-pixel variance weighting
  where available.
- New fields on Detection: `flux_err`, `flux_mjy_err`, `mag_ab_err`. Same
  fields added to Candidate from the best-SNR detection with uncertainty data.
- Report table includes Flux err and Mag err columns.
- DetailPanel shows uncertainties inline.
- DB schema updated; existing databases migrate automatically via ALTER TABLE.

## 1.1.1

- Fixed stdout redirection in acquisition.py: manual save/restore replaced with
  a context manager. Affected `_mast_download` and `_get_expected_filenames`.
- Fixed detection cache key missing `background_filter_size`.
- Fixed JSON crash when CatalogMatch `data` field is an empty string rather
  than null or a dict.
- Fixed silent migration failures in `_db.py`: unexpected ALTER TABLE errors
  now log warnings instead of being swallowed.
- Fixed InspectWindow cleanup race on rapid open/close cycles.
- Added `PRAGMA foreign_keys = ON` and WAL journal mode to all connections.

## 1.1.0

- `acquire()` and `reduce()` accept `ra`/`dec` kwargs for coordinate-based
  input.
- `get_fits_per_filter` uses WCS footprint containment to pick the right tile
  per filter.
- Downloads land directly in the slug dir; no obs-id subfolder.
- Coordinate slug format: `coord_{ra:.3f}_{dec:.3f}` with `-` -> `m`,
  `.` -> `p`.
- Coordinate-based acquire checks existing slug dirs before querying MAST.
- Cutout2D uses partial mode with NaN fill for edge sources.
- InspectWindow: two-row controls, labeled spinboxes, seven color palettes,
  per-filter alpha, Lupton RGB for all filter counts.
- GUI toolbar accepts decimal and sexagesimal coordinates.

## 1.0.0

- Initial stable release. Full pipeline: acquire, detect, resolve, report.
- GUI: SkyPanel, DetailPanel, InspectWindow, ReportsPanel, SettingsPanel.
- SQLite persistence with detection and catalog caching.
- SIMBAD, NED, and Gaia cross-reference.
- InspectWindow with per-filter cutouts, false-color Lupton composite,
  grayscale strip, R/G/B assignment, stretch/Q controls, save.
- Dockable panels with QSettings layout persistence.
- Collapsible log bar (SessionLogHandler, LogBar).
- Configurable background box size in SettingsPanel, folded into cache hash.
- Confidence score persisted and loaded from database.
- Inspection output files named with candidate ID prefix.
- Sky plot scroll-to-zoom; zoom buttons center on selected candidate.
- Reports panel right-click context menu.
- Partial download validation.
- Cross-platform file manager helpers (gui/platform.py).

## 0.9.x

- Removed "known-but-notable" classification; pipeline is two-tier: "known"
  and "unverified".
- Fixed `get_report()` reading from JSON exports instead of the database,
  causing known candidates to disappear on panel reload.
- InspectWindow: false-color Lupton composite, per-filter grayscale strip,
  channel controls, stretch/Q, save.
- Fixed numpy broadcast crash on low-SNR filter layers in composite.
- Floating color picker replaced with docked expanding picker.
- Per-filter alpha spinbox added.
- Additional color palettes: Stellar, Ionized Gas, Infrared.

## 0.8.x

- Dockable QDockWidget layout with QSettings persistence.
- Settings panel for key config values.
- Cross-platform file manager utility (gui/platform.py).
- View menu for reopening closed panels.

## 0.7.x

- Confidence scoring: SNR, filter count, catalog distance, flux quality.
- Report output into date subfolders with chart and cutout PNGs.
- SIMBAD canonical name resolution for folder naming.
- Progress bar with step counter.
- Sky plot scroll-to-zoom, zoom buttons, deselect on empty click.
- Sky panel clears after report deletion.

## 0.6.x

- MAST download stdout wrapped in devnull redirect.
- FILTER keyword reading regression fixed.
- SIMBAD slug handling: accounts for double-space Messier identifiers
  ("M  92" -> `m_92`) and "NAME" prefix on common names.
- Path separator normalization.
- Acquisition functions moved from survey.py to parallax/acquisition.py.
- Collapsible log bar with session history.
- Sky panel polish: y-axis label clip, deselect signal, zoom buttons.

## 0.5.x

- PyQt6 GUI module: MainWindow, RunWorker, SkyPanel, ReportsPanel,
  DetailPanel, ParallaxToolbar. `launch()` added to public API.
- `on_progress` callback on `reduce()`.
- Local file discovery in `acquire()` before querying MAST.
- Target name resolution with clear error on failure.

## 0.4.x

- Multi-filter detection across all available NIRCam filters; detections merged
  into unified candidates.
- Detection dataclass in `types.py`; detections list on Candidate.
- Report gains `filters` list and `report_inputs` table; drops `fits_path`
  and `observation_id`.
- i2d-only MAST filter; download count reduced significantly.
- `kron_flux` crash on degenerate sources fixed.
- Gaia failure flag propagation fixed.
- File discovery scoped per target to prevent cross-target collisions.

## 0.3.x

- Run fingerprinting: report IDs include a hash of input FITS files; re-runs
  overwrite rather than accumulate.
- Fingerprint computation moved inside `report()`.
- Report output into target subfolders with fingerprint-based filenames.
- Duplicate candidate row in report tables fixed.
- Note accumulation across runs fixed in `archive.annotate()`.

## 0.2.x

- Detection caching keyed on path, mtime, and detection parameters.
- Catalog cache with 30-day TTL.
- Candidate deduplication across runs.
- Batch catalog queries using vectorized sky matching.
- `clear_cache()` added to public API.

## 0.1.x

- Initial release: `acquire`, `detect`, `resolve`, `report`, `reduce`.
- MAST queries for JWST NIRCam i2d mosaics.
- Source extraction via photutils with Background2D.
- SIMBAD, NED, and Gaia cross-reference.
- SQLite persistence.
- MAST spatial query fixed (`coordinates` + `radius`; `s_fov` does not exist).
- SIMBAD updated for astroquery >= 0.4.8 TAP interface.
- FITS HDU selection: SCI extension preferred, fallback to first valid 2D HDU.
- Timezone handling fixed throughout; `_now_iso()` added to `types.py`.
