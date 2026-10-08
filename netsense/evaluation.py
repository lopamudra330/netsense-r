"""Event-level evaluation of degradation alarms against hidden ground truth.

Reusable by Experiments 2-4. Ground truth (onset, severity, episode type, cause) is used
ONLY here, after alarms have been produced. Episodes are identified by (seed, episode_id).

Definitions (docs/methodology.md, section 12.4):
- degrading phase: onset .. episode end (worsening) or .. last minute with severity > 0
  (recovering); stable episodes have none. Every other minute is healthy.
- an alarm event starting in a healthy minute is a FALSE ALARM; one starting inside the
  degrading phase is a VALID DETECTION. Pre-onset alarms never get early-detection credit.
"""

import numpy as np
import pandas as pd

from netsense import config as cfg

EPISODE_KEY = ["seed", "episode_id"]
VALID, ACTIVE_AT_ONSET, NO_ALARM = "valid_post_onset_detection", "alarm_active_at_onset", "no_valid_alarm"


def ground_truth(telemetry, episodes):
    """One row per episode: type, cause, minutes, onset, degrading-phase end, severe entry.

    `episodes` is the generator's episode table (needs seed, episode_id, onset_minute).
    """
    t = telemetry.sort_values(EPISODE_KEY + ["minute"])
    grouped = t.groupby(EPISODE_KEY, sort=True)
    truth = grouped.agg(episode_type=("episode_type", "first"), cause=("cause", "first"),
                        last_minute=("minute", "max")).reset_index()
    severe = (t[t.hidden_severity >= cfg.SEVERE_ENTRY_SEVERITY]
              .groupby(EPISODE_KEY)["minute"].min().rename("severe_entry"))
    last_positive = t[t.hidden_severity > 0].groupby(EPISODE_KEY)["minute"].max().rename("last_positive")
    truth = (truth.merge(episodes[EPISODE_KEY + ["onset_minute"]], on=EPISODE_KEY, how="left")
             .merge(severe, on=EPISODE_KEY, how="left").merge(last_positive, on=EPISODE_KEY, how="left"))
    truth["degrading_end"] = np.where(truth.episode_type == cfg.WORSENING, truth.last_minute,
                                      np.where(truth.episode_type == cfg.RECOVERING,
                                               truth.last_positive, np.nan))
    truth.loc[truth.episode_type == cfg.STABLE, "onset_minute"] = np.nan
    return truth.drop(columns="last_positive")


def _in_degrading_phase(minute, onset, end):
    return (~np.isnan(onset)) & (minute >= onset) & (minute <= end)


def classify_events(events, truth):
    """Label each alarm event 'false_alarm' or 'valid_detection' by its start minute."""
    merged = events.merge(truth[EPISODE_KEY + ["onset_minute", "degrading_end"]], on=EPISODE_KEY, how="left")
    inside = _in_degrading_phase(merged.start.to_numpy(float), merged.onset_minute.to_numpy(float),
                                 merged.degrading_end.to_numpy(float))
    return merged.assign(kind=np.where(inside, "valid_detection", "false_alarm"))


def episode_outcomes(classified, truth):
    """Per degrading episode: outcome category, detection minute, delay and severe lead."""
    degrading = truth[truth.episode_type != cfg.STABLE].copy()
    valid = (classified[classified.kind == "valid_detection"]
             .groupby(EPISODE_KEY)["start"].min().rename("detection_minute"))
    pre = classified[classified.kind == "false_alarm"].merge(
        degrading[EPISODE_KEY + ["onset_minute"]], on=EPISODE_KEY, how="inner", suffixes=("", "_t"))
    spans_onset = pre[(pre.start < pre.onset_minute) & (pre.online_end >= pre.onset_minute)]
    active = spans_onset[EPISODE_KEY].drop_duplicates().assign(alarm_active_at_onset=True)

    out = (degrading.merge(valid, on=EPISODE_KEY, how="left")
           .merge(active, on=EPISODE_KEY, how="left"))
    out["alarm_active_at_onset"] = out["alarm_active_at_onset"].astype("boolean").fillna(False).astype(bool)
    out["outcome"] = np.where(out.detection_minute.notna(), VALID,
                              np.where(out.alarm_active_at_onset, ACTIVE_AT_ONSET, NO_ALARM))
    out["delay_min"] = out.detection_minute - out.onset_minute
    out["severe_lead_min"] = out.severe_entry - out.detection_minute
    return out


def healthy_exposure(truth):
    """Healthy minutes per episode (every minute outside the degrading phase)."""
    total = truth.last_minute + 1
    degrading = (truth.degrading_end - truth.onset_minute + 1).fillna(0)
    return truth[EPISODE_KEY].assign(healthy_minutes=total - degrading)


def healthy_alarm_minutes(classified, truth):
    """Minutes of healthy time covered by an online alarm (start .. online_end), summed."""
    merged = classified.merge(truth[EPISODE_KEY + ["onset_minute", "degrading_end"]],
                              on=EPISODE_KEY, how="left", suffixes=("", "_t"))
    total = 0
    for event in merged.itertuples():
        minutes = np.arange(event.start, event.online_end + 1, dtype=float)
        total += int((~_in_degrading_phase(minutes, event.onset_minute, event.degrading_end)).sum())
    return total


def _quartiles(values):
    values = pd.Series(values).dropna()
    if values.empty:
        return np.nan, np.nan, np.nan
    return values.median(), values.quantile(0.25), values.quantile(0.75)


def summarise(classified, truth, group_cause=None):
    """Metric set for one detector (optionally restricted to degrading episodes of one cause)."""
    outcomes = episode_outcomes(classified, truth)
    if group_cause is not None:
        outcomes = outcomes[outcomes.cause == group_cause]
    worsening = outcomes[outcomes.episode_type == cfg.WORSENING]
    recovering = outcomes[outcomes.episode_type == cfg.RECOVERING]
    row = {
        "worsening_episodes": len(worsening),
        "recovering_episodes": len(recovering),
        "detection_rate_worsening_pct": 100 * (worsening.outcome == VALID).mean(),
        "detection_rate_recovering_pct": 100 * (recovering.outcome == VALID).mean(),
        "active_at_onset_only_pct": 100 * (outcomes.outcome == ACTIVE_AT_ONSET).mean(),
        "no_valid_alarm_pct": 100 * (outcomes.outcome == NO_ALARM).mean(),
        "any_alarm_active_at_onset_pct": 100 * outcomes.alarm_active_at_onset.mean(),
        "any_alarm_active_at_onset_n": int(outcomes.alarm_active_at_onset.sum()),
    }
    row["delay_median"], row["delay_p25"], row["delay_p75"] = _quartiles(outcomes.delay_min)
    detected_worsening = worsening[worsening.outcome == VALID]
    row["warned_before_severe_pct"] = 100 * (detected_worsening.severe_lead_min > 0).sum() / max(len(worsening), 1)
    row["severe_lead_median"], row["severe_lead_p25"], row["severe_lead_p75"] = \
        _quartiles(detected_worsening.severe_lead_min)
    if group_cause is None:
        exposure_min = healthy_exposure(truth).healthy_minutes.sum()
        false_alarms = int((classified.kind == "false_alarm").sum())
        row.update({"healthy_hours": exposure_min / 60, "false_alarms": false_alarms,
                    "false_alarms_per_100h": 100 * false_alarms / (exposure_min / 60),
                    "healthy_alarm_burden_pct": 100 * healthy_alarm_minutes(classified, truth) / exposure_min})
    return row


def first_trigger_by_cause(classified, truth):
    """Indicator group at the start of each episode's first valid detection, by true cause."""
    valid = classified[classified.kind == "valid_detection"].sort_values("start")
    first = valid.groupby(EPISODE_KEY).head(1).merge(truth[EPISODE_KEY + ["cause"]], on=EPISODE_KEY)
    return first.groupby(["cause", "trigger_group"]).size().rename("episodes").reset_index()
