import logging
import os
import yaml

logger = logging.getLogger(__name__)

_DEFAULTS = {
    "mast": {
        "instruments": ["NIRCAM", "MIRI"],
        "calib_level": 3,
        "product_type": "IMAGE",
    },
    "detection": {
        "snr_threshold": 3.0,
        "min_pixels": 5,
        "cutout_padding": 20,
        "kernel_fwhm": 2.0,
        "cutout_size": 60,
        "background_box_size": 50,
        "background_filter_size": 3,
    },
    "resolver": {
        "search_radius_arcsec": 2.0,
        "catalogs": ["SIMBAD", "NED", "GAIA"],
        "timeout_seconds": 30,
    },
    "report": {
        "output_format": "both",
        "include_known": False,
    },
    "cache": {
        "detection_enabled": True,
        "catalog_enabled": True,
        "catalog_ttl_days": 30,
        "candidate_match_radius_arcsec": 2.0,
    },
    "log": {
        "path": "data/parallax.log",
        "max_runs": 10,
    },
    "data": {
        "download_path": "data/downloads",
        "processed_path": "data/processed",
        "reports_path": "data/reports",
        "archive_path": "data/archive",
        "db_path": "data/parallax.db",
    },
}


def _deep_merge(base, override):
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


class Config:
    def __init__(self):
        self._data = {}
        self._loaded = False
        self._config_path = None

    def get(self, key: str, default=None):
        if not self._loaded:
            self.load()
        parts = key.split(".")
        val = self._data
        for p in parts:
            if isinstance(val, dict) and p in val:
                val = val[p]
            else:
                return default
        return val

    def set(self, key: str, value):
        parts = key.split(".")
        d = self._data
        for p in parts[:-1]:
            if p not in d or not isinstance(d[p], dict):
                d[p] = {}
            d = d[p]
        d[parts[-1]] = value

    def load(self, path: str | None = None):
        import copy
        self._data = copy.deepcopy(_DEFAULTS)

        if path is None:
            path = os.path.join(os.getcwd(), "config.yaml")

        self._config_path = path

        if os.path.isfile(path):
            with open(path, "r") as f:
                user = yaml.safe_load(f) or {}
            self._data = _deep_merge(self._data, user)
        else:
            logger.warning("config file not found at %s, using defaults", path)

        self._apply_env_overrides()
        self._loaded = True
        self._ensure_dirs()

        from parallax._db import init_db
        init_db()

    def _apply_env_overrides(self):
        prefix = "PARALLAX_"
        for k, v in os.environ.items():
            if not k.startswith(prefix):
                continue
            raw = k[len(prefix):]
            if "_" not in raw:
                continue
            section, key = raw.lower().split("_", 1)
            parsed = yaml.safe_load(v)
            if section not in self._data:
                self._data[section] = {}
            self._data[section][key] = parsed

    def _ensure_dirs(self):
        for k in ["download_path", "processed_path", "reports_path", "archive_path"]:
            p = self.get(f"data.{k}")
            if p:
                os.makedirs(p, exist_ok=True)
        archive = self.get("data.archive_path")
        if archive:
            os.makedirs(os.path.join(archive, "cutouts"), exist_ok=True)

    def save(self, path: str | None = None) -> None:
        if path is None:
            path = self._config_path or os.path.join(os.getcwd(), "config.yaml")
        with open(path, "w") as f:
            yaml.dump(self._data, f, default_flow_style=False, allow_unicode=True)

    # TODO: support config file in home directory as fallback


config = Config()
