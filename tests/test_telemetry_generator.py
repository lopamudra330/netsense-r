"""Tests for the synthetic telemetry generator.

These tests check the generator against its DOCUMENTED INTENT (schema, label rules,
episode definitions, mechanism directions). They deliberately do not test how easy
the states are to detect: that would turn the test suite into a tool for tuning data.
"""

import numpy as np
import pandas as pd
import pytest

from netsense import config as cfg
from netsense.telemetry_generator import (
    TELEMETRY_COLUMNS,
    build_episode_plan,
    burst_mask,
    generate_dataset,
    severity_to_state,
)


@pytest.fixture(scope="module")
def dataset():
    """Generate the default dataset once and share it across tests."""
    return generate_dataset(cfg.RANDOM_SEED)


@pytest.fixture(scope="module")
def telemetry(dataset):
    return dataset[0]


@pytest.fixture(scope="module")
def episodes(dataset):
    return dataset[1]


def per_episode_medians(telemetry, cause, column):
    """Median of `column` in NORMAL and SEVERE minutes, per worsening episode."""
    rows = telemetry[(telemetry.cause == cause) & (telemetry.episode_type == cfg.WORSENING)]
    medians = rows.groupby(["episode_id", "state"])[column].median().unstack()
    return medians[cfg.NORMAL], medians[cfg.SEVERE_DEGRADATION]


# 1. Schema
def test_schema_and_size(telemetry):
    assert list(telemetry.columns) == TELEMETRY_COLUMNS
    n_episodes = (cfg.N_STABLE_EPISODES + len(cfg.CAUSES)
                  * (cfg.N_WORSENING_EPISODES_PER_CAUSE + cfg.N_RECOVERING_EPISODES_PER_CAUSE))
    assert len(telemetry) == n_episodes * cfg.EPISODE_MINUTES
    assert pd.api.types.is_datetime64_any_dtype(telemetry.timestamp)
    for column in cfg.FEATURE_COLUMNS + ["hidden_severity"]:
        assert pd.api.types.is_float_dtype(telemetry[column]), column


# 2. Reproducibility
def test_same_seed_gives_identical_data(telemetry):
    again, _ = generate_dataset(cfg.RANDOM_SEED)
    pd.testing.assert_frame_equal(telemetry, again)


def test_different_seed_gives_different_data(telemetry):
    other, _ = generate_dataset(cfg.RANDOM_SEED + 1)
    assert not telemetry[cfg.FEATURE_COLUMNS].equals(other[cfg.FEATURE_COLUMNS])


# 3. Frozen study design
def test_episode_plan_counts():
    plan = build_episode_plan(np.random.default_rng(0))
    counts = pd.Series(plan).value_counts()
    assert counts[(cfg.STABLE, cfg.NO_CAUSE)] == 80
    for cause in cfg.CAUSES:
        assert counts[(cfg.WORSENING, cause)] == 60
        assert counts[(cfg.RECOVERING, cause)] == 30


# 4. Label convention at the band edges
@pytest.mark.parametrize("severity, expected", [
    (0.0, cfg.NORMAL),
    (0.0999, cfg.NORMAL),
    (0.10, cfg.EARLY_DEGRADATION),
    (0.3999, cfg.EARLY_DEGRADATION),
    (0.40, cfg.DEGRADED),
    (0.6999, cfg.DEGRADED),
    (0.70, cfg.SEVERE_DEGRADATION),
    (1.0, cfg.SEVERE_DEGRADATION),
])
def test_severity_band_edges(severity, expected):
    assert severity_to_state([severity])[0] == expected


# 5. Labels come only from hidden severity
def test_every_label_matches_its_severity_band(telemetry):
    assert (telemetry.state.to_numpy() == severity_to_state(telemetry.hidden_severity)).all()


# 6. Stable means stable
def test_stable_episodes_are_entirely_normal(telemetry):
    stable = telemetry[telemetry.episode_type == cfg.STABLE]
    assert (stable.hidden_severity == 0).all()
    assert (stable.state == cfg.NORMAL).all()


# 7. Episode-type definitions
def test_worsening_episodes_reach_severe_degradation(telemetry):
    worsening = telemetry[telemetry.episode_type == cfg.WORSENING]
    reached = worsening.groupby("episode_id").state.apply(
        lambda s: (s == cfg.SEVERE_DEGRADATION).any())
    assert reached.all()


def test_recovering_episodes_never_reach_severe_and_end_healthy(telemetry):
    recovering = telemetry[telemetry.episode_type == cfg.RECOVERING]
    assert not (recovering.state == cfg.SEVERE_DEGRADATION).any()
    final_minute = recovering[recovering.minute == cfg.EPISODE_MINUTES - 1]
    assert (final_minute.hidden_severity == 0).all()


# 8. Degradation onset is randomised rather than fixed across episodes
def test_onset_is_randomised_within_range(episodes):
    onsets = episodes.loc[episodes.episode_type != cfg.STABLE, "onset_minute"]
    assert onsets.between(*cfg.ONSET_RANGE_MIN).all()
    assert onsets.nunique() > 20


# 9. Physically valid clean data
def test_values_are_physically_valid(telemetry):
    assert not telemetry[cfg.FEATURE_COLUMNS].isna().any().any()
    assert (telemetry.latency_ms > 0).all()
    assert (telemetry.jitter_ms >= 0).all()
    for column in ["packet_loss_pct", "error_rate_pct", "retransmission_rate_pct"]:
        assert telemetry[column].between(0, 100).all(), column
    assert telemetry.throughput_mbps.between(0, cfg.LINK_CAPACITY_MBPS).all()


# 10. Documented congestion mechanism
def test_congestion_does_not_raise_errors_and_fills_the_link(telemetry):
    normal_err, severe_err = per_episode_medians(telemetry, cfg.CONGESTION, "error_rate_pct")
    ratio = severe_err / normal_err
    assert ratio.median() == pytest.approx(1.0, abs=0.25)

    normal_tp, severe_tp = per_episode_medians(telemetry, cfg.CONGESTION, "throughput_mbps")
    assert (severe_tp >= normal_tp).all()


# 11. Documented link-quality mechanism (including the accepted corrections)
def test_link_quality_raises_errors_not_average_latency(telemetry):
    normal_lat, severe_lat = per_episode_medians(telemetry, cfg.LINK_QUALITY, "latency_ms")
    assert (severe_lat - normal_lat).median() < 5.0  # "barely changes": a few ms

    normal_tp, severe_tp = per_episode_medians(telemetry, cfg.LINK_QUALITY, "throughput_mbps")
    assert (severe_tp < normal_tp).all()

    normal_err, severe_err = per_episode_medians(telemetry, cfg.LINK_QUALITY, "error_rate_pct")
    assert (severe_err > 10 * normal_err).all()


# 12. Common severity scale: about 20% loss at full severity for both causes
@pytest.mark.parametrize("cause", cfg.CAUSES)
def test_full_severity_gives_comparable_loss(telemetry, cause):
    full = telemetry[(telemetry.cause == cause) & (telemetry.hidden_severity >= 0.95)]
    assert 15.0 <= full.packet_loss_pct.mean() <= 25.0


# 13. Reactive status rule
def test_connection_state_follows_probe_loss_rule(telemetry):
    expected = np.where(telemetry.packet_loss_pct >= cfg.UNSTABLE_LOSS_PCT, "UNSTABLE", "UP")
    assert (telemetry.connection_state.to_numpy() == expected).all()


# 14. Leakage guard: detectors may only see observed telemetry
def test_feature_allow_list_excludes_ground_truth_and_bookkeeping(telemetry):
    forbidden = {"state", "hidden_severity", "cause", "episode_type", "episode_id",
                 "minute", "timestamp", "connection_state"}
    assert forbidden.isdisjoint(cfg.FEATURE_COLUMNS)
    assert set(cfg.FEATURE_COLUMNS) <= set(telemetry.columns)


# 15. Harmless transients behave as designed
def test_bursts_respect_duration_and_size_ranges():
    rng = np.random.default_rng(0)
    sizes, bursts = burst_mask(rng, cfg.TRAFFIC_BURST_START_CHANCE,
                               cfg.TRAFFIC_BURST_DURATION_MIN,
                               cfg.TRAFFIC_BURST_EXTRA_UTILISATION, n=10_000)
    assert len(bursts) > 50
    lo, hi = cfg.TRAFFIC_BURST_DURATION_MIN
    assert all(lo <= b["duration"] <= hi for b in bursts)
    low, high = cfg.TRAFFIC_BURST_EXTRA_UTILISATION
    assert all(low <= b["size"] <= high for b in bursts)
    assert ((sizes == 0) | ((sizes >= low) & (sizes <= high))).all()
