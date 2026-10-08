"""Tests for the Experiment 3 ML detector: causal features, isolation, threshold, events."""

import numpy as np
import pandas as pd
import pytest

from netsense import config as cfg
from netsense import evaluation as ev
from netsense import ml_detector as ml
from netsense.threshold_detector import alarm_events, rule_satisfied


def episode(seed, episode_id, n=40, latency=None, **labels):
    latency = np.arange(n, dtype=float) if latency is None else np.asarray(latency, dtype=float)
    frame = pd.DataFrame({"latency_ms": latency, "jitter_ms": 2.0, "error_rate_pct": 0.01,
                          "retransmission_rate_pct": 0.1, "packet_loss_pct": 0.0, "throughput_mbps": 50.0})
    defaults = {"episode_type": cfg.STABLE, "cause": cfg.NO_CAUSE, "state": cfg.NORMAL, "hidden_severity": 0.0}
    return frame.assign(seed=seed, episode_id=episode_id, minute=np.arange(n), **{**defaults, **labels})


# Causal features and warm-up
def test_features_match_hand_computed_values_and_warm_up_is_undefined():
    f = ml.build_features(episode(42, 0))  # latency = minute index
    assert f.loc[:8].isna().all().all()                       # minutes 0-8: warm-up
    assert f.loc[20, "latency_ms__current"] == 20
    assert f.loc[20, "latency_ms__mean10"] == pytest.approx(np.mean(np.arange(11, 21)))
    assert f.loc[20, "latency_ms__trend5"] == pytest.approx(np.mean(np.arange(16, 21)) - np.mean(np.arange(11, 16)))
    assert f.loc[20, "error_rate_pct__current"] == pytest.approx(np.log10(0.01 + cfg.ML_LOG_OFFSET))


def test_features_never_use_future_minutes():
    base = episode(42, 0)
    changed = base.copy()
    changed.loc[changed.minute > 25, ["latency_ms", "jitter_ms"]] = 999.0
    a, b = ml.build_features(base), ml.build_features(changed)
    pd.testing.assert_frame_equal(a.loc[:25], b.loc[:25])


def test_windows_reset_at_episode_and_seed_boundaries():
    a = episode(42, 7, latency=np.full(40, 10.0))
    b = episode(43, 7, latency=np.full(40, 500.0))   # same episode_id, different seed
    data = pd.concat([a, b], ignore_index=True)
    f = ml.build_features(data)
    second = f.iloc[40:]
    assert second.iloc[:9].isna().all().all()                 # warm-up again in the new episode
    assert (second["latency_ms__mean10"].dropna() == 500.0).all()   # never mixes the other seed


def test_timestamp_gap_is_not_bridged():
    data = episode(42, 0).drop(index=[20])                    # minute 20 missing
    f = ml.build_features(data)
    by_minute = f.set_axis(data.minute)
    assert by_minute.loc[21:29].isna().all().all()            # any window containing minute 20
    assert by_minute.loc[30].notna().all()                    # first complete window after the gap


# Hidden-label isolation
def test_feature_matrix_contains_only_declared_features_and_ignores_labels(generated):
    sample = generated[generated.episode_id < 10].assign(seed=42)
    altered = sample.assign(state=cfg.SEVERE_DEGRADATION, hidden_severity=1.0, cause=cfg.LINK_QUALITY,
                            episode_type=cfg.WORSENING, packet_loss_pct=99.0, throughput_mbps=1.0)
    a, b = ml.build_features(sample), ml.build_features(altered)
    assert list(a.columns) == ml.FEATURES and len(ml.FEATURES) == 12
    pd.testing.assert_frame_equal(a, b)


def test_labels_follow_declared_target():
    truth = pd.DataFrame([{"seed": 42, "episode_id": 0, "onset_minute": 10, "degrading_end": 30}])
    states = [cfg.NORMAL] * 12 + [cfg.EARLY_DEGRADATION] * 20 + [cfg.NORMAL] * 8
    data = episode(42, 0).assign(state=states)
    y = ml.training_labels(data, truth)
    assert (y.loc[:9] == 0).all()                            # before onset: healthy
    assert y.loc[10:11].isna().all()                          # after onset, still NORMAL: excluded
    assert (y.loc[12:30] == 1).all()                          # degrading
    assert (y.loc[31:] == 0).all()                            # after the degrading phase: healthy


# Model fitting, JSON reconstruction, determinism, training-only fitting
@pytest.fixture(scope="module")
def small_fit(generated):
    data = generated[generated.episode_id < 60].assign(seed=42)
    f = ml.build_features(data)
    y = (data.hidden_severity >= 0.1).astype(float)
    rows = f.notna().all(axis=1)
    return data, f, y, rows


def test_json_reconstruction_matches_fitted_model(small_fit):
    _, f, y, rows = small_fit
    model_dict, model = ml.fit(f[rows], y[rows])
    z = (f[rows].to_numpy() - np.array(model_dict["scaler_mean"])) / np.array(model_dict["scaler_std"])
    assert np.allclose(ml.score(f[rows], model_dict).to_numpy(), model.predict_proba(z)[:, 1], atol=1e-12)


def test_fitting_is_deterministic(small_fit):
    _, f, y, rows = small_fit
    assert ml.fit(f[rows], y[rows])[0] == ml.fit(f[rows], y[rows])[0]


def test_fit_uses_only_the_rows_given(small_fit):
    """Changing rows outside the fitting set (e.g. test episodes) changes nothing."""
    data, f, y, rows = small_fit
    fit_rows = rows & (data.episode_id < 40)
    other = f.copy()
    other.loc[data.episode_id >= 40] = 1e6
    pd.testing.assert_frame_equal(f[fit_rows], other[fit_rows])    # the fitting rows are identical
    a, b = ml.fit(f[fit_rows], y[fit_rows])[0], ml.fit(other[fit_rows], y[fit_rows])[0]
    for key in ("scaler_mean", "scaler_std", "coefficients"):         # equal up to summation order
        assert np.allclose(a[key], b[key], rtol=1e-10, atol=1e-12)
    assert a["intercept"] == pytest.approx(b["intercept"], rel=1e-10)


# Threshold
def test_threshold_rule_closest_share_and_strict_comparison():
    scores = np.arange(1, 101) / 100.0                         # 100 distinct scores
    tau, achieved = ml.derive_threshold(scores, target_rate=0.03)
    assert tau == pytest.approx(0.97) and achieved == pytest.approx(0.03)
    assert (scores > tau).mean() == pytest.approx(0.03)


def test_threshold_ties_choose_larger_score():
    scores = np.array([0.1] * 50 + [0.2] * 50)                 # shares above: 0.5 (s=0.1), 0.0 (s=0.2)
    tau, achieved = ml.derive_threshold(scores, target_rate=0.25)
    assert tau == 0.2 and achieved == 0.0


# Persistence and events reuse Experiment 2's frozen functions
def test_ml_events_reuse_frozen_event_logic():
    pattern = [0] * 10 + [1, 1, 1] + [0] * 10
    data = episode(42, 0, n=len(pattern))
    events = ml.ml_events(data, np.array(pattern, dtype=bool), 3, 3)
    expected = alarm_events(rule_satisfied(np.array(pattern, dtype=bool), 3, 3), cfg.ALARM_CLEAR_MINUTES)
    assert events.drop(columns=["seed", "episode_id", "trigger_group"]).to_dict("records") == expected


def test_ml_events_keep_seed_episode_identity_separate():
    a, b = episode(42, 7, n=20), episode(43, 7, n=20)
    data = pd.concat([a, b], ignore_index=True)
    flags = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    events = ml.ml_events(data, flags, 1, 1)
    assert list(zip(events.seed, events.episode_id)) == [(42, 7)]


def test_evaluate_refuses_without_intact_frozen_state_and_never_loads_data(monkeypatch):
    from experiments import e2_threshold_baseline, e3_ml_detector
    def forbidden(*args, **kwargs):
        raise AssertionError("held-out data must not be loaded")
    monkeypatch.setattr(e2_threshold_baseline, "test_data", forbidden)
    monkeypatch.setattr(e3_ml_detector, "FROZEN_COMMIT", "0000000")  # not the frozen commit
    with pytest.raises(SystemExit):
        e3_ml_detector.evaluate()
