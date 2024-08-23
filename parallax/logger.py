import os
import re
import tempfile
import time
from datetime import datetime, UTC


class RunLogger:
    """Rolling run log for pipeline scripts. Not used by the library itself."""

    def __init__(self, log_path, max_runs=10):
        self._path = log_path
        self._max = max_runs
        self._start_time = None
        self._lines = []

        d = os.path.dirname(log_path)
        if d:
            os.makedirs(d, exist_ok=True)

    def start(self, version, target):
        self._start_time = time.monotonic()
        ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S")
        self._lines = [f"=== Run {ts} | v{version} | {target} ==="]

    def step(self, name, message, cache_hit=None, elapsed=None):
        t = time.monotonic() - self._start_time
        tag = f"[{t:05.1f}]"

        suffix = ""
        if cache_hit is not None:
            if cache_hit and elapsed is not None:
                suffix = f"  [CACHE HIT {elapsed:.1f}s]"
            elif cache_hit:
                suffix = "  [CACHE HIT]"
        elif elapsed is not None:
            m, s = divmod(elapsed, 60)
            if m >= 1:
                suffix = f"  [{int(m)}m {s:.0f}s]"
            else:
                suffix = f"  [{s:.1f}s]"

        self._lines.append(f"{tag} {name:<18}{message}{suffix}")

    def end(self):
        t = time.monotonic() - self._start_time
        self._lines.append(f"[{t:05.1f}] {'done':<18}total {t:.1f}s")
        self._lines.append("=== End ===")
        self._flush()

    def _flush(self):
        existing = ""
        if os.path.isfile(self._path):
            with open(self._path, "r") as f:
                existing = f.read()

        new_block = "\n".join(self._lines) + "\n"
        full = existing + new_block

        blocks = re.findall(
            r"=== Run .*?=== End ===\n?",
            full, re.DOTALL
        )
        kept = blocks[-self._max:]

        d = os.path.dirname(self._path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".log.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                for b in kept:
                    if not b.endswith("\n"):
                        b += "\n"
                    f.write(b)
            # TODO: handle windows rename-over-existing edge case
            if os.path.exists(self._path):
                os.replace(tmp, self._path)
            else:
                os.rename(tmp, self._path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
