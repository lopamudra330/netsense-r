"""Tests for event-level evaluation against hidden ground truth (made-up episodes)."""

import numpy as np
import pandas as pd
import pytest

from netsense import config as cfg
from netsense import evaluation as ev


def truth_row(seed, episode_id, episode_type, onset, degrading_end, severe, cause=cfg.CONGESTION,
              last_minute=99):
    return {"seed": seed, "episode_id": episode_id, "episode_type": episode_type, "cause": cause,
            "last_minute": last_minute, "onset_minute": onset, "degrading_end": degrading_end,
            "severe_entry": severe}


def events(*rows):
    cols = ["seed", "episode_id", "start", "effective_end", "confirmed_clear", "online_end", "trigger_group"]
    return pd.DataFrame([dict(zip(cols, r)) for r in rows], columns=cols)


TRUTH = pd.DataFrame([
    truth_row(42, 0, cfg.WORSENING, 40, 99, 80),
    truth_row(42, 1, cfg.RECOVERING, 40, 70, np.nan),
    truth_row(42, 2, cfg.STABLE, np.nan, np.nan, np.nan, cause=cfg.NO_CAUSE),
])


def test_events_classified_by_start_minute():
    e = events((42, 0, 39, 45, 50, 50, "delay"),   # starts 1 min before onset -> false alarm
               (42, 0, 40, 45, 50, 50, "delay"),   # starts at onset -> valid
               (42, 1, 71, 72, 77, 77, "error"),   # after full recovery -> false alarm
               (42, 2, 10, 12, 17, 17, "delay"))   # stable -> false alarm
    kinds = ev.classify_events(e, TRUTH).kind.tolist()
    assert kinds == ["false_alarm", "valid_detection", "false_alarm", "false_alarm"]


def test_one_long_false_alarm_counts_once():
    e = events((42, 2, 0, 98, None, 99, "delay"))
    row = ev.summarise(ev.classify_events(e, TRUTH), TRUTH)
    assert row["false_alarms"] == 1


def test_delay_and_lead_arithmetic_including_negative_lead():
    e = events((42, 0, 52, 60, 65, 65, "delay"), (42, 1, 45, 50, 55, 55, "error"))
    out = ev.episode_outcomes(ev.classify_events(e, TRUTH), TRUTH).set_index("episode_id")
    assert out.loc[0, "delay_min"] == 12 and out.loc[0, "severe_lead_min"] == 28
    late = events((42, 0, 85, 90, 95, 95, "delay"))
    out = ev.episode_outcomes(ev.classify_events(late, TRUTH), TRUTH).set_index("episode_id")
    assert out.loc[0, "severe_lead_min"] == -5  # detected after severe entry: kept negative


def test_alarm_active_at_onset_is_not_detection_and_not_silence():
    pre = events((42, 0, 30, 99, None, 99, "delay"))  # starts before onset, never clears
    out = ev.episode_outcomes(ev.classify_events(pre, TRUTH), TRUTH).set_index("episode_id")
    assert out.loc[0, "outcome"] == ev.ACTIVE_AT_ONSET
    assert np.isnan(out.loc[0, "delay_min"])            # no zero/negative delay credit
    assert out.loc[1, "outcome"] == ev.NO_ALARM


def test_pre_onset_alarm_then_new_post_onset_alarm_counts_as_valid():
    e = events((42, 0, 30, 35, 40, 40, "delay"), (42, 0, 50, 60, 65, 65, "delay"))
    out = ev.episode_outcomes(ev.classify_events(e, TRUTH), TRUTH).set_index("episode_id")
    assert out.loc[0, "outcome"] == ev.VALID and out.loc[0, "delay_min"] == 10
    assert out.loc[0, "alarm_active_at_onset"]  # contextual flag still recorded


def test_false_alarm_rate_and_burden_use_healthy_time_only():
    # healthy minutes: ep0 40, ep1 40 + 29, ep2 100 -> 209 minutes
    e = events((42, 2, 10, 14, 19, 19, "delay"),      # 10 healthy minutes online
               (42, 0, 38, 45, 50, 50, "delay"))      # 2 healthy minutes (38, 39) online
    row = ev.summarise(ev.classify_events(e, TRUTH), TRUTH)
    assert row["healthy_hours"] == pytest.approx(209 / 60)
    assert row["false_alarms"] == 2
    assert row["false_alarms_per_100h"] == pytest.approx(100 * 2 / (209 / 60))
    assert row["healthy_alarm_burden_pct"] == pytest.approx(100 * 12 / 209)


def test_seed_and_episode_id_identify_episodes_together():
    truth = pd.concat([TRUTH, TRUTH.assign(seed=43)], ignore_index=True)
    e = events((43, 0, 50, 55, 60, 60, "delay"))
    out = ev.episode_outcomes(ev.classify_events(e, truth), truth).set_index(["seed", "episode_id"])
    assert out.loc[(43, 0), "outcome"] == ev.VALID
    assert out.loc[(42, 0), "outcome"] == ev.NO_ALARM
