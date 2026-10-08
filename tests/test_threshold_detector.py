"""Tests for the engineering detector: thresholds, persistence, causal alarm events."""

import numpy as np
import pandas as pd
import pytest

from netsense import config as cfg
from netsense.threshold_detector import (alarm_events, derive_thresholds, rule_satisfied,
                                         run_detector, suspicious_minutes)
from experiments.e2_threshold_baseline import select_training

THRESHOLDS = {"latency_ms": 50.0, "jitter_ms": 10.0, "error_rate_pct": 0.1,
              "retransmission_rate_pct": 1.0}


def episode(seed, episode_id, latency, **labels):
    n = len(latency)
    frame = pd.DataFrame({"latency_ms": latency, "jitter_ms": 1.0, "error_rate_pct": 0.01,
                          "retransmission_rate_pct": 0.1, "packet_loss_pct": 0.0,
                          "throughput_mbps": 50.0})
    defaults = {"episode_type": cfg.STABLE, "cause": cfg.NO_CAUSE, "state": cfg.NORMAL,
                "hidden_severity": 0.0}
    return frame.assign(seed=seed, episode_id=episode_id, minute=np.arange(n), **{**defaults, **labels})


# Threshold application
def test_condition_is_strictly_above_threshold_and_any_indicator_counts():
    rows = pd.DataFrame({"latency_ms": [50.0, 50.1, 1.0], "jitter_ms": [1.0, 1.0, 10.5],
                         "error_rate_pct": [0.0, 0.0, 0.0], "retransmission_rate_pct": [0.0, 0.0, np.nan]})
    flags = suspicious_minutes(rows, THRESHOLDS)
    assert flags.any(axis=1).tolist() == [False, True, True]  # equal is not above; NaN is not


def test_thresholds_use_only_normal_training_minutes():
    normal = episode(42, 0, np.arange(100.0))
    degraded = episode(42, 1, np.full(100, 1e6), state=cfg.DEGRADED)
    thresholds, used = derive_thresholds(pd.concat([normal, degraded]))
    assert used == 100
    assert thresholds["latency_ms"] == pytest.approx(np.percentile(np.arange(100.0), 99))


# Persistence
def test_consecutive_rule():
    pattern = [1, 1, 0, 1, 1, 1, 1]
    assert rule_satisfied(pattern, 3, 3).tolist() == [False, False, False, False, False, True, True]


def test_k_of_w_rule_including_short_window_at_episode_start():
    pattern = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    # minute 2: 2 of the first 3 -> not >= 3; add a third hit at minute 4
    pattern[4] = 1
    # hits at minutes 0, 2, 4, 11: three within the last 10 minutes at minutes 4-9 and 11
    expected = [False] * 4 + [True] * 6 + [False, True, False]
    assert rule_satisfied(pattern, 3, 10).tolist() == expected
    assert rule_satisfied([1, 1, 1], 3, 10).tolist() == [False, False, True]


# Causal alarm events
def test_alarm_cannot_close_before_five_quiet_minutes():
    satisfied = [True, False, False, False, False]  # only 4 quiet minutes so far
    events = alarm_events(satisfied, clear_minutes=5)
    assert len(events) == 1 and events[0]["confirmed_clear"] is None  # still open, not closed
    events = alarm_events(satisfied + [False], clear_minutes=5)
    assert events[0]["confirmed_clear"] == 5 and events[0]["effective_end"] == 0


def test_closure_decision_never_depends_on_future_minutes():
    """Events confirmed by minute t are identical whatever happens after t."""
    base = [True, True, False, False, False, False, False, True, False]
    a = alarm_events(base + [True] * 5)
    b = alarm_events(base + [False] * 5)
    closed_by_6 = lambda evs: [e for e in evs if e["confirmed_clear"] is not None and e["confirmed_clear"] <= 6]
    assert closed_by_6(a) == closed_by_6(b) == [
        {"start": 0, "effective_end": 1, "confirmed_clear": 6, "online_end": 6}]


def test_retrigger_before_confirmation_continues_the_same_event():
    satisfied = [True, False, False, False, True, False, False, False, False, False]
    events = alarm_events(satisfied)
    assert len(events) == 1
    assert events[0] == {"start": 0, "effective_end": 4, "confirmed_clear": 9, "online_end": 9}


def test_continuous_alarm_is_one_event_and_closes_at_episode_end():
    events = alarm_events([False] + [True] * 8)
    assert events == [{"start": 1, "effective_end": 8, "confirmed_clear": None, "online_end": 8}]


# Episode boundaries and identity
def test_persistence_does_not_carry_across_episodes_or_seeds():
    end_of_first = episode(42, 7, [1.0] * 8 + [99.0] * 2)       # 2 suspicious minutes at the end
    start_of_next = episode(43, 7, [99.0] * 1 + [1.0] * 9)      # 1 at the start, same episode_id
    events = run_detector(pd.concat([end_of_first, start_of_next]), THRESHOLDS, k=3, w=3)
    assert events.empty  # 2 + 1 would satisfy 3-in-a-row only if counts leaked across


def test_same_episode_id_in_different_seeds_has_separate_alarm_histories():
    a = episode(42, 7, [99.0] * 10)
    b = episode(43, 7, [1.0] * 10)
    events = run_detector(pd.concat([a, b]), THRESHOLDS, k=1, w=1)
    assert list(zip(events.seed, events.episode_id)) == [(42, 7)]


# No hidden-label leakage; determinism
def test_alarms_ignore_hidden_labels(generated):
    sample = generated[generated.episode_id < 20].assign(seed=42)
    altered = sample.assign(state=cfg.SEVERE_DEGRADATION, hidden_severity=1.0,
                            cause=cfg.LINK_QUALITY, episode_type=cfg.WORSENING,
                            packet_loss_pct=99.0, throughput_mbps=1.0, connection_state="UNSTABLE")
    thresholds = {c: float(sample[c].quantile(0.95)) for c in cfg.DETECTOR_INDICATORS}
    for k, w in cfg.DETECTORS.values():
        pd.testing.assert_frame_equal(run_detector(sample, thresholds, k, w),
                                      run_detector(altered, thresholds, k, w))


def test_detector_is_deterministic(generated):
    sample = generated[generated.episode_id < 20].assign(seed=42)
    thresholds = {c: float(sample[c].quantile(0.95)) for c in cfg.DETECTOR_INDICATORS}
    pd.testing.assert_frame_equal(run_detector(sample, thresholds, 3, 10),
                                  run_detector(sample, thresholds, 3, 10))


# Train/test isolation
def test_training_selection_and_thresholds_ignore_test_telemetry(generated):
    from netsense.splits import split_episodes
    episodes = generated[["episode_id"]].drop_duplicates().assign(onset_minute=60)
    split = split_episodes(generated, cfg.RANDOM_SEED)
    test_ids = set(split.loc[split.split == "test", "episode_id"])
    corrupted = generated.copy()
    is_test = corrupted.episode_id.isin(test_ids)
    corrupted.loc[is_test, cfg.FEATURE_COLUMNS] = 1e9  # garbage in every test telemetry value

    clean_train, _ = select_training(generated, episodes, cfg.RANDOM_SEED)
    dirty_train, _ = select_training(corrupted, episodes, cfg.RANDOM_SEED)
    pd.testing.assert_frame_equal(clean_train, dirty_train)
    assert not set(clean_train.episode_id) & test_ids
    assert derive_thresholds(clean_train) == derive_thresholds(dirty_train)
