import json
import logging
import threading
from datetime import datetime, UTC

from parallax._db import get_db
from parallax.types import Criteria, _watch_id, _now_iso

logger = logging.getLogger(__name__)

_poll_thread = None
_stop_event = threading.Event()


def _criteria_from_dict(d: dict) -> Criteria:
    return Criteria(
        name=d["name"],
        instruments=d["instruments"],
        filters=d.get("filters"),
        ra=d.get("ra"),
        dec=d.get("dec"),
        radius_deg=d.get("radius_deg"),
        check_interval_minutes=d.get("check_interval_minutes", 60),
    )


def watch(criteria: Criteria | dict) -> str:
    """Register a watch for new MAST data matching criteria."""
    if isinstance(criteria, dict):
        criteria = _criteria_from_dict(criteria)

    if not criteria.name:
        raise ValueError("watch name is required")
    if not criteria.instruments:
        raise ValueError("instruments list cannot be empty")

    wid = _watch_id()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO watches (id, name, criteria, last_checked, n_hits, active) "
            "VALUES (?,?,?,?,?,?)",
            (wid, criteria.name, json.dumps({
                "name": criteria.name,
                "instruments": criteria.instruments,
                "filters": criteria.filters,
                "ra": criteria.ra,
                "dec": criteria.dec,
                "radius_deg": criteria.radius_deg,
                "check_interval_minutes": criteria.check_interval_minutes,
            }), None, 0, 1)
        )
    return wid


def unwatch(watch_id: str) -> None:
    """Remove a registered watch."""
    with get_db() as conn:
        row = conn.execute("SELECT id FROM watches WHERE id = ?",
                           (watch_id,)).fetchone()
        if row is None:
            raise KeyError(watch_id)
        conn.execute("DELETE FROM watch_hits WHERE watch_id = ?", (watch_id,))
        conn.execute("DELETE FROM watches WHERE id = ?", (watch_id,))


def status() -> list[dict]:
    """Return status of all registered watches."""
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM watches").fetchall()
    result = []
    for r in rows:
        result.append({
            "watch_id": r["id"],
            "name": r["name"],
            "criteria": json.loads(r["criteria"]),
            "last_checked": r["last_checked"],
            "n_hits": r["n_hits"],
            "active": bool(r["active"]),
        })
    return result


def _check_watch(watch_id, criteria_dict, last_checked):
    from astroquery.mast import Observations

    kwargs = {
        "instrument_name": criteria_dict["instruments"],
        "calib_level": 3,
        "dataproduct_type": "IMAGE",
    }
    if criteria_dict.get("filters"):
        kwargs["filter_name"] = criteria_dict["filters"]
    if last_checked:
        from astropy.time import Time
        t = Time(last_checked, format="isot", scale="utc")
        kwargs["t_min"] = [t.mjd, None]
    if criteria_dict.get("ra") is not None:
        kwargs["s_ra"] = criteria_dict["ra"]
        kwargs["s_dec"] = criteria_dict["dec"]
        kwargs["s_fov"] = criteria_dict.get("radius_deg", 0.1)

    results = Observations.query_criteria(**kwargs)
    if results is None or len(results) == 0:
        return []

    obs_ids = [str(r["obs_id"]) for r in results]

    with get_db() as conn:
        existing = conn.execute(
            "SELECT observation_id FROM watch_hits WHERE watch_id = ?",
            (watch_id,)
        ).fetchall()
    seen = {r["observation_id"] for r in existing}

    new_ids = [oid for oid in obs_ids if oid not in seen]

    if new_ids:
        now = _now_iso()
        with get_db() as conn:
            for oid in new_ids:
                conn.execute(
                    "INSERT INTO watch_hits (watch_id, observation_id, detected_at) "
                    "VALUES (?,?,?)", (watch_id, oid, now)
                )
            conn.execute(
                "UPDATE watches SET n_hits = n_hits + ?, last_checked = ? WHERE id = ?",
                (len(new_ids), now, watch_id)
            )
    else:
        now = _now_iso()
        with get_db() as conn:
            conn.execute("UPDATE watches SET last_checked = ? WHERE id = ?",
                         (now, watch_id))

    return new_ids


def check(watch_id: str | None = None) -> dict[str, list[str]]:
    """Manually trigger a MAST check for one or all active watches."""
    results = {}

    if watch_id is not None:
        with get_db() as conn:
            row = conn.execute("SELECT * FROM watches WHERE id = ?",
                               (watch_id,)).fetchone()
        if row is None:
            raise KeyError(watch_id)

        criteria = json.loads(row["criteria"])
        new = _check_watch(watch_id, criteria, row["last_checked"])
        results[watch_id] = new
    else:
        watches = status()
        for w in watches:
            if not w["active"]:
                continue
            try:
                new = _check_watch(w["watch_id"], w["criteria"], w["last_checked"])
                results[w["watch_id"]] = new
            except Exception as e:
                logger.warning("check failed for %s: %s", w["watch_id"], e)
                results[w["watch_id"]] = []

    return results


def _poll_loop(interval_minutes):
    while not _stop_event.is_set():
        watches = status()
        now = datetime.now(UTC)
        for w in watches:
            if not w["active"]:
                continue
            interval = interval_minutes or w["criteria"].get("check_interval_minutes", 60)
            last = w["last_checked"]
            if last is not None:
                last_dt = datetime.fromisoformat(last)
                elapsed = (now - last_dt).total_seconds() / 60
                if elapsed < interval:
                    continue
            try:
                check(w["watch_id"])
            except Exception as e:
                logger.warning("watch %s poll failed: %s", w["watch_id"], e)

        _stop_event.wait(timeout=60)


def start(interval_minutes: int | None = None) -> None:
    """Start background polling in a daemon thread."""
    global _poll_thread
    if _poll_thread is not None and _poll_thread.is_alive():
        logger.warning("polling already running")
        return

    _stop_event.clear()
    _poll_thread = threading.Thread(target=_poll_loop, args=(interval_minutes,),
                                   daemon=True)
    _poll_thread.start()


def stop() -> None:
    """Stop background polling if running."""
    global _poll_thread
    if _poll_thread is None or not _poll_thread.is_alive():
        return
    _stop_event.set()
    _poll_thread.join(timeout=5)
    _poll_thread = None
