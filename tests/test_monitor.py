import time
from unittest.mock import patch, MagicMock

import pytest

from parallax.types import Criteria


class TestWatch:
    def test_watch_and_status(self, tmp_db):
        from parallax import monitor
        crit = Criteria(name="test watch", instruments=["NIRCAM"])
        wid = monitor.watch(crit)
        assert wid.startswith("watch_")

        st = monitor.status()
        assert len(st) == 1
        assert st[0]["name"] == "test watch"
        assert st[0]["active"] is True

    def test_watch_from_dict(self, tmp_db):
        from parallax import monitor
        wid = monitor.watch({"name": "dict watch", "instruments": ["MIRI"]})
        st = monitor.status()
        assert any(s["watch_id"] == wid for s in st)

    def test_watch_no_name_raises(self, tmp_db):
        from parallax import monitor
        with pytest.raises(ValueError):
            monitor.watch({"name": "", "instruments": ["NIRCAM"]})

    def test_watch_no_instruments_raises(self, tmp_db):
        from parallax import monitor
        with pytest.raises(ValueError):
            monitor.watch({"name": "bad", "instruments": []})


class TestUnwatch:
    def test_unwatch_removes(self, tmp_db):
        from parallax import monitor
        wid = monitor.watch(Criteria(name="temp", instruments=["NIRCAM"]))
        monitor.unwatch(wid)
        st = monitor.status()
        assert not any(s["watch_id"] == wid for s in st)

    def test_unwatch_missing(self, tmp_db):
        from parallax import monitor
        with pytest.raises(KeyError):
            monitor.unwatch("watch_nonexist")


class TestCheck:
    @patch("parallax.monitor._check_watch")
    def test_returns_new_obs(self, mock_cw, tmp_db):
        from parallax import monitor
        wid = monitor.watch(Criteria(name="chk", instruments=["NIRCAM"]))
        mock_cw.return_value = ["obs_001", "obs_002"]

        result = monitor.check(wid)
        assert wid in result
        assert result[wid] == ["obs_001", "obs_002"]

    def test_unknown_watch_raises(self, tmp_db):
        from parallax import monitor
        with pytest.raises(KeyError):
            monitor.check("watch_fake")


class TestStartStop:
    def test_start_stop(self, tmp_db):
        from parallax import monitor
        monitor.watch(Criteria(name="bg", instruments=["NIRCAM"]))

        with patch("parallax.monitor._check_watch", return_value=[]):
            monitor.start(interval_minutes=1)
            time.sleep(0.2)
            monitor.stop()

    def test_double_start(self, tmp_db):
        from parallax import monitor
        with patch("parallax.monitor._check_watch", return_value=[]):
            monitor.start(interval_minutes=1)
            monitor.start(interval_minutes=1)  # should not raise
            time.sleep(0.1)
            monitor.stop()
