from contextlib import contextmanager
import logging
import sqlite3
from parallax.config import config

logger = logging.getLogger(__name__)


@contextmanager
def get_db():
    path = config.get("data.db_path")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """Create schema if not present."""
    # TODO: add migration support for schema changes
    with get_db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        for stmt in [
            "ALTER TABLE reports ADD COLUMN fingerprint TEXT",
            "ALTER TABLE reports ADD COLUMN filters TEXT",
            "ALTER TABLE candidates ADD COLUMN confidence REAL NOT NULL DEFAULT 0.0",
            "ALTER TABLE candidates ADD COLUMN flux_err REAL",
            "ALTER TABLE candidates ADD COLUMN flux_mjy_err REAL",
            "ALTER TABLE candidates ADD COLUMN mag_ab_err REAL",
            "ALTER TABLE candidate_detections ADD COLUMN flux_mjy REAL",
            "ALTER TABLE candidate_detections ADD COLUMN mag_ab REAL",
            "ALTER TABLE candidate_detections ADD COLUMN flux_err REAL",
            "ALTER TABLE candidate_detections ADD COLUMN flux_mjy_err REAL",
            "ALTER TABLE candidate_detections ADD COLUMN mag_ab_err REAL",
            "ALTER TABLE candidates ADD COLUMN hints TEXT NOT NULL DEFAULT '[]'",
            "ALTER TABLE candidate_detections ADD COLUMN local_rms REAL",
        ]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                if "duplicate column" in str(e).lower():
                    pass
                else:
                    logger.warning("migration failed: %s -- %s", stmt, e)
        try:
            conn.execute(
                "UPDATE candidates SET classification = 'known' "
                "WHERE classification = 'known-but-notable'"
            )
        except sqlite3.OperationalError as e:
            logger.warning("classification migration: %s", e)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    flux REAL,
    snr REAL,
    classification TEXT NOT NULL DEFAULT 'unverified',
    report_id TEXT NOT NULL,
    pixel_x REAL,
    pixel_y REAL,
    created_at TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '[]',
    notes TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    flux_err REAL,
    flux_mjy_err REAL,
    mag_ab_err REAL,
    hints TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS catalog_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    catalog TEXT NOT NULL,
    source_id TEXT,
    separation_arcsec REAL,
    object_type TEXT,
    redshift REAL,
    data TEXT
);

CREATE TABLE IF NOT EXISTS candidate_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    timestamp TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT
);

CREATE TABLE IF NOT EXISTS reports (
    id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    instrument TEXT,
    filter TEXT,
    observation_id TEXT,
    fits_path TEXT,
    created_at TEXT NOT NULL,
    n_sources_detected INTEGER,
    n_catalog_matched INTEGER,
    n_unverified INTEGER,
    json_path TEXT,
    md_path TEXT,
    fingerprint TEXT
);

CREATE TABLE IF NOT EXISTS watches (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    criteria TEXT NOT NULL,
    last_checked TEXT,
    n_hits INTEGER NOT NULL DEFAULT 0,
    active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS watch_hits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    watch_id TEXT NOT NULL REFERENCES watches(id),
    observation_id TEXT NOT NULL,
    detected_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS detection_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fits_path TEXT NOT NULL,
    fits_hash TEXT NOT NULL,
    snr_threshold REAL NOT NULL,
    min_pixels INTEGER NOT NULL,
    kernel_fwhm REAL NOT NULL,
    detections TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(fits_hash, snr_threshold, min_pixels, kernel_fwhm)
);

CREATE TABLE IF NOT EXISTS catalog_cache (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    field_key TEXT NOT NULL UNIQUE,
    catalog TEXT NOT NULL,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    radius_arcsec REAL NOT NULL,
    results TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS report_inputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT NOT NULL REFERENCES reports(id),
    fits_path TEXT NOT NULL,
    observation_id TEXT,
    filter TEXT
);

CREATE INDEX IF NOT EXISTS idx_report_inputs_report
    ON report_inputs(report_id);

CREATE TABLE IF NOT EXISTS candidate_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    candidate_id TEXT NOT NULL REFERENCES candidates(id),
    filter TEXT NOT NULL,
    flux REAL,
    snr REAL,
    pixel_x REAL,
    pixel_y REAL,
    flux_mjy REAL,
    mag_ab REAL,
    flux_err REAL,
    flux_mjy_err REAL,
    mag_ab_err REAL,
    local_rms REAL
);

CREATE INDEX IF NOT EXISTS idx_candidate_detections_candidate
    ON candidate_detections(candidate_id);

CREATE INDEX IF NOT EXISTS idx_candidates_report ON candidates(report_id);
CREATE INDEX IF NOT EXISTS idx_candidates_classification ON candidates(classification);
CREATE INDEX IF NOT EXISTS idx_candidates_coords ON candidates(ra, dec);
CREATE INDEX IF NOT EXISTS idx_catalog_matches_candidate ON catalog_matches(candidate_id);
CREATE INDEX IF NOT EXISTS idx_history_candidate ON candidate_history(candidate_id);
CREATE INDEX IF NOT EXISTS idx_detection_cache_hash
    ON detection_cache(fits_hash, snr_threshold, min_pixels, kernel_fwhm);
CREATE INDEX IF NOT EXISTS idx_catalog_cache_key
    ON catalog_cache(field_key);
CREATE INDEX IF NOT EXISTS idx_catalog_cache_expires
    ON catalog_cache(expires_at);
"""
