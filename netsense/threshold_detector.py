"""Engineering monitoring baseline: thresholds, persistence rules and alarm events.

The detector reads ONLY the four indicator columns in cfg.DETECTOR_INDICATORS, plus the
(seed, episode_id, minute) keys that order observations. Hidden labels (state, severity,
cause, episode type) never enter any function here except `derive_thresholds`, which uses
the NORMAL label of TRAINING rows to describe healthy behaviour.
"""

import numpy as np
import pandas as pd

from netsense import config as cfg

EPISODE_KEY = ["seed", "episode_id"]
INDICATORS = cfg.DETECTOR_INDICATORS


def derive_thresholds(training, percentile=cfg.DETECTOR_THRESHOLD_PERCENTILE):
    """Upper threshold per indicator: the given percentile of NORMAL training minutes.

    Returns (thresholds dict, number of NORMAL rows used).
    """
    normal = training.loc[training.state == cfg.NORMAL, INDICATORS]
    thresholds = {c: float(np.percentile(normal[c].dropna(), percentile)) for c in INDICATORS}
    return thresholds, len(normal)


def suspicious_minutes(telemetry, thresholds):
    """Per-row DataFrame of booleans: which indicators are strictly above threshold.

    Missing values count as not suspicious.
    """
    return pd.DataFrame({c: telemetry[c].to_numpy() > thresholds[c] for c in INDICATORS},
                        index=telemetry.index)


def rule_satisfied(suspicious, k, w):
    """True at minute t if at least k of the last w minutes (t-w+1 .. t) are suspicious.

    `suspicious` is a boolean array over consecutive minutes of ONE episode. Uses only the
    current and past minutes. At the start of an episode the window is shorter than w.
    """
    counts = np.convolve(np.asarray(suspicious, dtype=int), np.ones(w, dtype=int))[: len(suspicious)]
    return counts >= k


def alarm_events(satisfied, clear_minutes=cfg.ALARM_CLEAR_MINUTES):
    """Turn a per-minute rule array into alarm events, causally.

    An alarm starts when the rule is satisfied while no alarm is active. Closure is confirmed
    only on the `clear_minutes`-th consecutive unsatisfied minute; if the rule is satisfied
    again before that, the same event continues. An event open at the end is closed there.

    Returns a list of dicts: start, effective_end (last satisfied minute),
    confirmed_clear (minute closure was confirmed, or None if still open at the end),
    online_end (last minute the alarm was shown online).
    """
    events, active, start, last_satisfied, quiet = [], False, 0, 0, 0
    for t, ok in enumerate(satisfied):
        if not active:
            if ok:
                active, start, last_satisfied, quiet = True, t, t, 0
            continue
        if ok:
            last_satisfied, quiet = t, 0
        else:
            quiet += 1
            if quiet == clear_minutes:
                events.append({"start": start, "effective_end": last_satisfied,
                               "confirmed_clear": t, "online_end": t})
                active = False
    if active:
        events.append({"start": start, "effective_end": last_satisfied,
                       "confirmed_clear": None, "online_end": len(satisfied) - 1})
    return events


def _episode_minutes(episode, column_frame):
    """Values indexed by every minute 0..last minute; missing minutes are absent (NaN)."""
    minutes = episode["minute"].to_numpy()
    full = np.arange(minutes.max() + 1)
    return column_frame.set_axis(minutes).reindex(full)


def run_detector(telemetry, thresholds, k, w, clear_minutes=cfg.ALARM_CLEAR_MINUTES):
    """Apply one detector to every episode. Returns one row per alarm event.

    Columns: seed, episode_id, start, effective_end, confirmed_clear, online_end,
    trigger_group (indicator group suspicious at the start minute: delay/error/both).
    """
    telemetry = telemetry.reset_index(drop=True)  # row labels may repeat across seeds
    flags = suspicious_minutes(telemetry, thresholds)
    rows = []
    ordered = telemetry[EPISODE_KEY + ["minute"]].sort_values(EPISODE_KEY + ["minute"])
    for (seed, episode_id), episode in ordered.groupby(EPISODE_KEY, sort=True):
        per_minute = _episode_minutes(episode, flags.loc[episode.index]).fillna(False).astype(bool)
        any_suspicious = per_minute.any(axis=1).to_numpy()  # a missing minute is not suspicious
        for event in alarm_events(rule_satisfied(any_suspicious, k, w), clear_minutes):
            at_start = per_minute.iloc[event["start"]]
            groups = {cfg.INDICATOR_GROUPS[c] for c in INDICATORS if at_start[c]}
            rows.append({"seed": seed, "episode_id": episode_id, **event,
                         "trigger_group": {0: "none", 1: next(iter(groups), "none"), 2: "both"}[len(groups)]})
    return pd.DataFrame(rows, columns=EPISODE_KEY + ["start", "effective_end", "confirmed_clear",
                                                     "online_end", "trigger_group"])
