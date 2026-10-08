"""Experiment 3: interpretable logistic-regression degradation detector.

Features are built ONLY from the four observable indicators in cfg.ML_INDICATORS.
Hidden labels (state, severity, cause, episode type) are used only to build training
targets (`training_labels`) and never enter the feature matrix.

The output is a *logistic model score* (the predicted probability under the fitted logistic
model). Probability calibration is not performed.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from netsense import config as cfg
from netsense.threshold_detector import alarm_events, rule_satisfied

EPISODE_KEY = ["seed", "episode_id"]
FEATURES = [f"{c}__{kind}" for c in cfg.ML_INDICATORS for kind in ("current", "mean10", "trend5")]


def transformed(telemetry):
    """The four indicators after the fixed transform (log10(x + offset) for the two rates)."""
    values = telemetry[cfg.ML_INDICATORS].astype(float).copy()
    for c in cfg.ML_LOG_INDICATORS:
        values[c] = np.log10(values[c] + cfg.ML_LOG_OFFSET)
    return values


def build_features(telemetry):
    """Causal features per row, computed within each (seed, episode_id) on the full minute index.

    A feature at minute t uses only minutes t-9..t. If any of those minutes is missing for
    any indicator (gap or warm-up), all features at t are NaN. Returns a DataFrame aligned
    with `telemetry.index` with the columns in FEATURES.
    """
    w, half = cfg.ML_WINDOW_MIN, cfg.ML_TREND_HALF_MIN
    values = transformed(telemetry)
    out = pd.DataFrame(np.nan, index=telemetry.index, columns=FEATURES)
    for _, episode in telemetry.groupby(EPISODE_KEY, sort=False):
        minutes = episode["minute"].to_numpy()
        full = pd.RangeIndex(0, minutes.max() + 1)
        series = values.loc[episode.index].set_axis(minutes).reindex(full)  # gaps -> NaN
        complete = series.notna().all(axis=1).astype(int).rolling(w, min_periods=w).sum() == w
        feats = {}
        for c in cfg.ML_INDICATORS:
            x = series[c]
            recent = x.rolling(half, min_periods=half).mean()
            feats[f"{c}__current"] = x
            feats[f"{c}__mean10"] = x.rolling(w, min_periods=w).mean()
            feats[f"{c}__trend5"] = recent - recent.shift(half)
        frame = pd.DataFrame(feats)[FEATURES].where(complete)  # undefined unless full window
        out.loc[episode.index, FEATURES] = frame.loc[minutes].to_numpy()
    return out


def training_labels(telemetry, truth):
    """1 = degrading phase with state EARLY/DEGRADED/SEVERE; 0 = healthy; NaN = excluded.

    Excluded: inside the degrading phase while the hidden state is still NORMAL
    (severity below 0.10). Evaluation-only ground truth; never a feature.
    """
    phase = telemetry[EPISODE_KEY].merge(truth[EPISODE_KEY + ["onset_minute", "degrading_end"]],
                                         on=EPISODE_KEY, how="left")
    minute = telemetry["minute"].to_numpy()
    inside = (minute >= phase.onset_minute.to_numpy()) & (minute <= phase.degrading_end.to_numpy())
    degraded_state = telemetry["state"].to_numpy() != cfg.NORMAL
    labels = np.where(~inside, 0.0, np.where(degraded_state, 1.0, np.nan))
    return pd.Series(labels, index=telemetry.index)


def fit(features, labels, columns=FEATURES):
    """Standardise on the given rows, fit logistic regression. Returns a plain-number model dict."""
    X = features[columns].to_numpy(float)
    mean, std = X.mean(axis=0), X.std(axis=0)
    model = LogisticRegression(max_iter=cfg.ML_MAX_ITER)
    model.fit((X - mean) / std, labels.to_numpy(int))
    return {"features": list(columns), "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
            "coefficients": model.coef_[0].tolist(), "intercept": float(model.intercept_[0])}, model


def score(features, model_dict):
    """Logistic model score reconstructed from stored numbers (no refitting).

    Rows with undefined features get NaN.
    """
    X = features[model_dict["features"]].to_numpy(float)
    z = (X - np.array(model_dict["scaler_mean"])) / np.array(model_dict["scaler_std"])
    logit = z @ np.array(model_dict["coefficients"]) + model_dict["intercept"]
    return pd.Series(1.0 / (1.0 + np.exp(-logit)), index=features.index)


def derive_threshold(healthy_scores, target_rate=cfg.ML_TARGET_HEALTHY_RATE):
    """tau: the distinct healthy score s whose share of healthy scores strictly above s is
    closest to target_rate; equally close -> the larger s. Returns (tau, achieved rate)."""
    values = np.sort(np.asarray(healthy_scores, dtype=float))
    unique = np.unique(values)
    above = 1.0 - np.searchsorted(values, unique, side="right") / len(values)  # share > s
    distance = np.abs(above - target_rate)
    best = np.flatnonzero(distance == distance.min()).max()  # tie -> larger s
    return float(unique[best]), float(above[best])


def ml_events(telemetry, suspicious, k, w, clear_minutes=cfg.ALARM_CLEAR_MINUTES):
    """Alarm events from a per-row suspicious flag, via Experiment 2's frozen functions.

    Missing minutes count as not suspicious. Output columns match run_detector().
    """
    rows = []
    flags = pd.Series(np.asarray(suspicious, dtype=bool), index=telemetry.index)
    ordered = telemetry[EPISODE_KEY + ["minute"]].sort_values(EPISODE_KEY + ["minute"])
    for (seed, episode_id), episode in ordered.groupby(EPISODE_KEY, sort=True):
        minutes = episode["minute"].to_numpy()
        per_minute = np.zeros(minutes.max() + 1, dtype=bool)
        per_minute[minutes] = flags.loc[episode.index].to_numpy()
        for event in alarm_events(rule_satisfied(per_minute, k, w), clear_minutes):
            rows.append({"seed": seed, "episode_id": episode_id, **event, "trigger_group": "ml"})
    return pd.DataFrame(rows, columns=EPISODE_KEY + ["start", "effective_end", "confirmed_clear",
                                                     "online_end", "trigger_group"])
