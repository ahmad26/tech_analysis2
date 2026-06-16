import json
import time

import pytest

from src.alert_tracker import AlertTracker


def test_new_alert_not_duplicate(tmp_path):
    tracker = AlertTracker(str(tmp_path / "state.json"))
    assert not tracker.is_duplicate("key1")


def test_recorded_alert_is_duplicate(tmp_path):
    tracker = AlertTracker(str(tmp_path / "state.json"))
    tracker.record("key1")
    assert tracker.is_duplicate("key1")


def test_state_persists(tmp_path):
    state_file = str(tmp_path / "state.json")
    tracker1 = AlertTracker(state_file)
    tracker1.record("key1")

    tracker2 = AlertTracker(state_file)
    assert tracker2.is_duplicate("key1")


def test_cleanup_removes_old_entries(tmp_path):
    state_file = str(tmp_path / "state.json")
    tracker = AlertTracker(state_file, ttl_hours=1)

    # Manually insert an old entry
    old_time = time.time() - 7200  # 2 hours ago
    tracker._seen["old_key"] = old_time
    tracker._seen["new_key"] = time.time()
    tracker._save()

    tracker.cleanup()
    assert not tracker.is_duplicate("old_key")
    assert tracker.is_duplicate("new_key")


def test_load_corrupt_file(tmp_path):
    state_file = tmp_path / "state.json"
    state_file.write_text("not valid json")

    tracker = AlertTracker(str(state_file))
    assert not tracker.is_duplicate("anything")


def test_multiple_records(tmp_path):
    tracker = AlertTracker(str(tmp_path / "state.json"))
    tracker.record("key1")
    tracker.record("key2")
    tracker.record("key3")

    assert tracker.is_duplicate("key1")
    assert tracker.is_duplicate("key2")
    assert tracker.is_duplicate("key3")
    assert not tracker.is_duplicate("key4")
