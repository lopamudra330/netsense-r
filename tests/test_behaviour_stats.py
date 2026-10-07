"""Tests for Experiment 1's descriptive statistics, on small made-up data."""

import numpy as np
import pandas as pd
import pytest

from netsense import config as cfg
from netsense.behaviour_stats import (baseline_relative, outside_reference_share,
                                      probability_of_superiority, rising_summaries,
                                      rising_window, standardised_difference)

rng = np.random.default_rng(0)


def episode(seed, episode_id, episode_type, severity, value=10.0, cause=cfg.CONGESTION):
    n = len(severity)
    frame = pd.DataFrame({m: np.full(n, value) for m in cfg.FEATURE_COLUMNS})
    return frame.assign(seed=seed, episode_id=episode_id, minute=np.arange(n),
                        episode_type=episode_type, cause=cause,
                        hidden_severity=np.asarray(severity, dtype=float))


# 4. Reference-range share means what it says
def test_reference_share_is_about_two_percent_for_identical_data():
    normal, same = rng.normal(0, 1, 200_000), rng.normal(0, 1, 200_000)
    assert outside_reference_share(normal, same) == pytest.approx(2.0, abs=0.2)


def test_reference_share_is_about_all_for_clearly_shifted_data():
    assert outside_reference_share(rng.normal(0, 1, 10_000), rng.normal(10, 1, 10_000)) > 99.9


# 5. Standardised difference: magnitude and direction
def test_standardised_difference_known_shift():
    normal = rng.normal(5, 2, 200_000)
    assert standardised_difference(normal, normal + 4) == pytest.approx(2.0, abs=0.02)
    assert standardised_difference(normal, normal - 4) == pytest.approx(-2.0, abs=0.02)


# 6. Baseline ratio uses only the episode's own data
def test_baseline_ratio_is_one_for_constant_episode_and_not_mixed_between_links():
    data = pd.concat([episode(42, 0, cfg.STABLE, np.zeros(120), value=10.0),
                      episode(42, 1, cfg.STABLE, np.zeros(120), value=50.0)])
    relative = baseline_relative(data)
    assert np.allclose(relative[cfg.FEATURE_COLUMNS].to_numpy(), 1.0)


def test_baseline_of_zero_gives_undefined_ratio_not_infinity():
    data = episode(42, 0, cfg.STABLE, np.zeros(120), value=0.0)
    assert baseline_relative(data)[cfg.FEATURE_COLUMNS].isna().all().all()


# 7. Rising window: same rule for both episode types; stable excluded
def test_rising_window_runs_from_onset_to_first_minute_at_end_severity():
    severity = np.r_[np.zeros(60), np.linspace(0.01, 0.6, 60), np.zeros(240)]
    data = pd.concat([episode(42, 0, cfg.WORSENING, severity),
                      episode(42, 1, cfg.RECOVERING, severity),
                      episode(42, 2, cfg.STABLE, np.zeros(360))])
    window = rising_window(data)
    first_at_end = 60 + int(np.argmax(np.linspace(0.01, 0.6, 60) >= 0.25))
    for episode_id in (0, 1):
        minutes = window.loc[window.episode_id == episode_id, "minute"]
        assert minutes.min() == 60 and minutes.max() == first_at_end  # inclusive
    assert 2 not in set(window.episode_id)


def test_episode_that_never_reaches_end_severity_is_excluded():
    severity = np.r_[np.zeros(60), np.full(60, 0.2), np.zeros(240)]
    assert rising_window(episode(42, 0, cfg.RECOVERING, severity)).empty


# Multi-seed identity: (seed, episode_id), never episode_id alone
def test_same_episode_id_from_different_seeds_stays_distinct():
    severity = np.r_[np.zeros(60), np.linspace(0.01, 0.6, 60), np.zeros(240)]
    data = pd.concat([episode(42, 7, cfg.WORSENING, severity, value=10.0),
                      episode(43, 7, cfg.WORSENING, severity, value=99.0)])
    relative = baseline_relative(data)
    assert np.allclose(relative[cfg.FEATURE_COLUMNS].to_numpy(), 1.0)  # own baselines
    summaries = rising_summaries(rising_window(relative))
    assert len(summaries) == 2
    assert set(zip(summaries.seed, summaries.episode_id)) == {(42, 7), (43, 7)}


# 8. Probability of superiority
def test_probability_of_superiority():
    a = rng.normal(0, 1, 2000)
    assert probability_of_superiority(a, a) == pytest.approx(0.5)
    assert probability_of_superiority([5, 6, 7], [1, 2, 3]) == 1.0
    assert probability_of_superiority([1, 2, 3], [5, 6, 7]) == 0.0
    assert probability_of_superiority([1, 1], [1, 1]) == 0.5  # ties count as half
