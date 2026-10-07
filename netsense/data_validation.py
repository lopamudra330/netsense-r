"""Telemetry Integrity & Validation Layer ("Trust Layer").

Before asking a model to learn from telemetry, can we establish that the observations
themselves are structurally and operationally trustworthy?

Every rule here uses only knowledge an operator would really have (link capacity,
sampling interval, the status rule, allowed values). No rule reads the study labels'
values (`state`, `hidden_severity`) to decide whether telemetry is valid.

Levels
------
FAIL  structurally impossible, or record integrity broken -> row is quarantined
WARN  incomplete or operationally inconsistent, but not impossible -> kept, reported
INFO  statistically unusual but potentially legitimate -> kept, never affects status

PASS means only that no defined integrity/validation rule found a problem.
It does not certify that the telemetry is objectively correct.
"""

import numpy as np
import pandas as pd

from netsense import config as cfg
from netsense.telemetry_generator import TELEMETRY_COLUMNS

FAIL, WARN, INFO = "FAIL", "WARN", "INFO"

TELEMETRY = cfg.FEATURE_COLUMNS
KEYS = cfg.RECORD_KEY_COLUMNS
STATUS = cfg.STATUS_COLUMN
LABELS = cfg.STUDY_LABEL_COLUMNS

# Row-level checks: id -> (level, columns needed, short name, explanation)
CHECKS = {
    "S2-telemetry": (FAIL, KEYS + TELEMETRY, "Missing key or detector telemetry",
                     "The observation cannot be placed in time or is missing a value the analysis needs"),
    "S2-status": (WARN, [STATUS], "Missing connection status",
                  "Status unavailable; status-based checks skip this minute"),
    "S2-labels": (FAIL, LABELS, "Missing study label",
                  "Study-label integrity: the row cannot be evaluated (synthetic-study field)"),
    "S3-status": (WARN, [STATUS], "Invalid connection status",
                  "Status value outside UP/UNSTABLE, e.g. a vendor or version format difference"),
    "S3-labels": (FAIL, ["episode_type", "cause", "state"], "Invalid study label",
                  "Study-label integrity: label value outside the allowed set"),
    "S4": (FAIL, TELEMETRY, "Negative telemetry value",
           "Delay, jitter, throughput and rates cannot be negative"),
    "S5": (FAIL, cfg.PERCENT_COLUMNS, "Percentage above 100",
           "A share of packets, frames or segments cannot exceed 100%"),
    "S6": (FAIL, ["throughput_mbps"], "Throughput above link capacity",
           "Delivered traffic cannot exceed the configured link capacity"),
    "I1": (FAIL, ["episode_id", "minute"], "Duplicate episode/minute",
           "The same observation key appears more than once"),
    "I2": (FAIL, ["episode_id", "timestamp"], "Duplicate timestamp",
           "The same timestamp appears more than once within an episode"),
    "I3": (FAIL, ["episode_id", "timestamp"], "Timestamp out of sequence",
           "A timestamp is earlier than the previous record's in the same episode"),
    "C1": (WARN, ["episode_id", "timestamp"], "Timestamp gap",
           "Observations are missing between this record and the previous one"),
    "O1": (WARN, [STATUS, "packet_loss_pct"], "Status inconsistent with loss",
           "connection_state disagrees with the rule UNSTABLE <=> probe loss >= 5%"),
    "O2": (WARN, ["episode_id", "timestamp", "latency_ms", "jitter_ms"], "Frozen probe metrics",
           "Latency and jitter repeat exactly, as from a stuck poller"),
    "O3": (WARN, ["error_rate_pct", "throughput_mbps"], "Saturated error counter",
           "Every frame reported as failing while traffic is still delivered"),
    "U1": (INFO, TELEMETRY, "Statistically unusual value",
           "Far from the bulk of the data; kept, NOT evidence of bad data"),
}

# Dataset-level schema checks (S1): id -> (level, short name)
SCHEMA_CHECKS = {
    "S1-missing-required": (FAIL, "Required column missing"),
    "S1-missing-status": (WARN, "Status column missing"),
    "S1-unexpected": (WARN, "Unexpected extra column"),
}


# --- Individual rules ---------------------------------------------------------

def _sorted_within_episode(df):
    """Row order sorted by episode then timestamp (stable, keeps original index)."""
    return df.sort_values(["episode_id", "timestamp"], kind="stable")


def _missing_key_or_telemetry(df):
    return df[KEYS + TELEMETRY].isna().any(axis=1)


def _missing_status(df):
    return df[STATUS].isna()


def _missing_label(df):
    return df[LABELS].isna().any(axis=1)


def _invalid_status(df):
    return df[STATUS].notna() & ~df[STATUS].isin(cfg.CONNECTION_STATES)


def _invalid_label(df):
    allowed = {"episode_type": cfg.EPISODE_TYPES,
               "cause": cfg.CAUSES + [cfg.NO_CAUSE],
               "state": cfg.STATES}
    bad = pd.Series(False, index=df.index)
    for column, values in allowed.items():
        bad |= df[column].notna() & ~df[column].isin(values)
    return bad


def _negative(df):
    return (df[TELEMETRY] < 0).any(axis=1)


def _percent_above_100(df):
    return (df[cfg.PERCENT_COLUMNS] > 100).any(axis=1)


def _above_capacity(df):
    return df["throughput_mbps"] > cfg.LINK_CAPACITY_MBPS


def _duplicate_key(df):
    return df.duplicated(["episode_id", "minute"], keep=False)


def _duplicate_timestamp(df):
    return df.duplicated(["episode_id", "timestamp"], keep=False)


def _out_of_sequence(df):
    step = df.groupby("episode_id")["timestamp"].diff()  # file order
    return step < pd.Timedelta(0)


def _gap(df):
    ordered = _sorted_within_episode(df.dropna(subset=["episode_id", "timestamp"]))
    step = ordered.groupby("episode_id")["timestamp"].diff()
    gap_rows = step[step > pd.Timedelta(seconds=cfg.SAMPLE_INTERVAL_SECONDS)].index
    return pd.Series(df.index.isin(gap_rows), index=df.index)


def _status_inconsistent(df):
    valid = df[STATUS].isin(cfg.CONNECTION_STATES) & df["packet_loss_pct"].notna()
    unstable_by_rule = df["packet_loss_pct"] >= cfg.UNSTABLE_LOSS_PCT
    return valid & ((df[STATUS] == "UNSTABLE") != unstable_by_rule)


def _frozen_probe_metrics(df):
    ordered = _sorted_within_episode(df)
    grouped = ordered.groupby("episode_id")
    same = ((ordered["latency_ms"] == grouped["latency_ms"].shift())
            & (ordered["jitter_ms"] == grouped["jitter_ms"].shift()))
    run_id = (~same).cumsum()                       # a new run starts where values change
    run_length = run_id.map(run_id.value_counts())
    frozen = run_length >= cfg.FROZEN_RUN_MINUTES
    return frozen.reindex(df.index)


def _saturated_error_counter(df):
    return (df["error_rate_pct"] >= 100) & (df["throughput_mbps"] > 0)


def _statistically_unusual(df):
    values = df[TELEMETRY]
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    limit = q3 + cfg.UNUSUAL_IQR_FACTOR * (q3 - q1)
    return (values > limit).any(axis=1)


RULES = {
    "S2-telemetry": _missing_key_or_telemetry, "S2-status": _missing_status,
    "S2-labels": _missing_label, "S3-status": _invalid_status, "S3-labels": _invalid_label,
    "S4": _negative, "S5": _percent_above_100, "S6": _above_capacity,
    "I1": _duplicate_key, "I2": _duplicate_timestamp, "I3": _out_of_sequence,
    "C1": _gap, "O1": _status_inconsistent, "O2": _frozen_probe_metrics,
    "O3": _saturated_error_counter, "U1": _statistically_unusual,
}


# --- Running the layer ----------------------------------------------------------

def prepare(telemetry):
    """Return a copy with a fresh 0..n-1 index and parsed timestamps."""
    df = telemetry.reset_index(drop=True).copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def schema_problems(df):
    """Dataset-level S1 checks. Returns {check id: list of column names}."""
    required = KEYS + TELEMETRY + LABELS
    return {
        "S1-missing-required": [c for c in required if c not in df.columns],
        "S1-missing-status": [] if STATUS in df.columns else [STATUS],
        "S1-unexpected": [c for c in df.columns if c not in TELEMETRY_COLUMNS],
    }


def run_checks(telemetry):
    """Apply every row-level rule.

    Returns (prepared DataFrame, flags DataFrame). Flags has one boolean column per
    check; a check whose columns are missing is filled with NA ("not evaluated").
    """
    df = prepare(telemetry)
    flags = pd.DataFrame(index=df.index)
    for check_id, rule in RULES.items():
        needed = CHECKS[check_id][1]
        if all(c in df.columns for c in needed):
            flags[check_id] = rule(df).fillna(False).astype(bool)
        else:
            flags[check_id] = pd.NA
    return df, flags


def _example(df, rows, check_id):
    if len(rows) == 0:
        return ""
    row = df.loc[rows[0]]
    columns = [c for c in CHECKS[check_id][1] if c not in ("episode_id", "minute")]
    values = ", ".join(f"{c}={row[c]}" for c in columns[:3] if c in df.columns)
    return f"ep {row.get('episode_id')}, min {row.get('minute')}: {values}"


def build_report(df, flags):
    """One row per check: level, status, affected count, episodes, example, explanation."""
    rows = []
    for check_id, columns in schema_problems(df).items():
        level, name = SCHEMA_CHECKS[check_id]
        rows.append({"check": check_id, "name": name, "level": level,
                     "status": level.lower() if columns else "pass",
                     "affected": len(columns), "episodes_affected": "",
                     "example": ", ".join(columns),
                     "explanation": "Dataset structure differs from the expected schema"})
    for check_id, (level, _, name, explanation) in CHECKS.items():
        column = flags[check_id]
        if column.isna().all():
            rows.append({"check": check_id, "name": name, "level": level,
                         "status": "not evaluated", "affected": 0, "episodes_affected": "",
                         "example": "", "explanation": "Required columns are missing"})
            continue
        hit = column.astype(bool)
        flagged = df.index[hit]
        episodes = df.loc[hit, "episode_id"].dropna().unique() if "episode_id" in df else []
        listed = ", ".join(str(int(e)) for e in sorted(episodes)[:5])
        rows.append({"check": check_id, "name": name, "level": level,
                     "status": level.lower() if hit.any() else "pass",
                     "affected": int(hit.sum()),
                     "episodes_affected": f"{len(episodes)} ({listed})" if len(episodes) else "0",
                     "example": _example(df, flagged, check_id),
                     "explanation": explanation})
    return pd.DataFrame(rows)


def overall_status(report):
    """FAIL if any FAIL-level finding, else WARN if any WARN finding, else PASS.

    INFO findings never change the status. PASS means only that no defined rule found
    a problem; it does not certify that the telemetry is correct.
    """
    if (report["status"] == "fail").any():
        return "FAIL"
    if (report["status"] == "warn").any():
        return "WARN"
    return "PASS"


def validate(telemetry):
    """Run the whole layer. Returns (prepared df, flags, report, overall status)."""
    df, flags = run_checks(telemetry)
    report = build_report(df, flags)
    return df, flags, report, overall_status(report)


def split_validated(df, flags):
    """Separate validated telemetry from quarantined rows.

    - any FAIL-level flag -> quarantined;
    - exact duplicates: the first copy is kept, further copies are quarantined;
    - conflicting duplicates (same key, different values): every copy is quarantined;
    - WARN and INFO rows are kept; gaps are left as gaps (nothing is filled in).

    Returns (validated telemetry, quarantined rows with a `reasons` column).
    """
    fail_ids = [c for c, (level, *_) in CHECKS.items()
                if level == FAIL and c not in ("I1", "I2") and flags[c].notna().all()]
    reasons = pd.Series("", index=df.index)
    for check_id in fail_ids:
        reasons[flags[check_id].astype(bool)] += check_id + " "

    extra_copy = df.duplicated(keep="first")
    reasons[extra_copy] += "exact-duplicate-copy "
    distinct = df.drop_duplicates()
    for key in (["episode_id", "minute"], ["episode_id", "timestamp"]):
        conflicting = distinct[distinct.duplicated(key, keep=False)][key].drop_duplicates()
        in_conflict = df.set_index(key).index.isin(conflicting.set_index(key).index)
        reasons[in_conflict] += "conflicting-duplicate "

    quarantine = reasons != ""
    quarantined = df[quarantine].assign(reasons=reasons[quarantine].str.strip())
    return df[~quarantine], quarantined
