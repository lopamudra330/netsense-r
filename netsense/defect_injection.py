"""Deliberate, documented corruption of a COPY of generated telemetry.

Used only to evaluate the Telemetry Integrity & Validation Layer against a known
ground truth (the defect manifest). The generated input is never modified.

Each defect event affects a span of minutes in one episode (its "footprint").
Footprints are kept apart so that every validation flag can be traced to one event.
"""

import numpy as np
import pandas as pd

from netsense import config as cfg
from netsense.data_validation import CHECKS, INFO

# defect type -> (expected check, realistic cause)
DEFECT_TYPES = {
    "missing_value": ("S2-telemetry", "Poll timeout or counter read failure"),
    "exact_duplicate": ("I1", "Collector retry or at-least-once delivery"),
    "conflicting_duplicate": ("I1", "Two collectors reporting the same minute"),
    "negative_latency": ("S4", "Measurement or timestamp-processing error"),
    "negative_throughput": ("S4", "Counter reset producing a negative difference"),
    "percent_over_100": ("S5", "Unit-conversion error applied twice"),
    "throughput_over_capacity": ("S6", "32-bit byte-counter wrap over one poll interval"),
    "timestamp_gap": ("C1", "Collector or network outage"),
    "out_of_sequence": ("I3", "Delayed or buffered record arrival"),
    "invalid_category": ("S3-status", "Status format differs between vendors or versions"),
    "inconsistent_status": ("O1", "Status taken from a different poll cycle"),
    "frozen_value": ("O2", "Stuck poller repeating its last value"),
    "plausible_offset": ("none", "Miscalibrated probe; plausible, undetectable by design"),
}

def _text(value):
    """Plain text for the manifest: numbers as shortest round-trip decimals."""
    if isinstance(value, str):
        return value
    return repr(float(value))


MANIFEST_COLUMNS = ["defect_id", "defect_type", "episode_id", "minute_start", "minute_end",
                    "field", "original_value", "injected_value", "expected_check",
                    "detectable"]


def _footprint_length(defect_type, rng):
    """Minutes covered by one event (drawn now so spacing can be checked)."""
    if defect_type == "timestamp_gap":
        removed = int(rng.integers(cfg.GAP_LENGTH_RANGE_MIN[0], cfg.GAP_LENGTH_RANGE_MIN[1] + 1))
        return removed + 1  # removed minutes + the first minute after the gap
    if defect_type == "out_of_sequence":
        return 2
    if defect_type == "frozen_value":
        return cfg.FROZEN_DEFECT_LENGTH_MIN + 1  # source minute + repeated minutes
    return 1


def _choose_locations(generated, rng):
    """Pick non-overlapping (type, episode, start, length) events, deterministically."""
    used = {}  # episode -> list of (start, end)
    events = []
    eligible_loss = generated["packet_loss_pct"] >= cfg.PERCENT_OVER_100_MIN_LOSS
    for defect_type in DEFECT_TYPES:
        placed = 0
        while placed < cfg.DEFECT_EVENTS_PER_TYPE:
            length = _footprint_length(defect_type, rng)
            candidates = generated.index[eligible_loss] if defect_type == "percent_over_100" \
                else generated.index
            row = generated.loc[rng.choice(candidates)]
            episode, start = int(row.episode_id), int(row.minute)
            end = start + length - 1
            if (start < cfg.DEFECT_EDGE_MARGIN_MIN
                    or end > cfg.EPISODE_MINUTES - 1 - cfg.DEFECT_EDGE_MARGIN_MIN):
                continue
            if any(start - cfg.DEFECT_SPACING_MIN <= e and s - cfg.DEFECT_SPACING_MIN <= end
                   for s, e in used.get(episode, [])):
                continue
            used.setdefault(episode, []).append((start, end))
            events.append((defect_type, episode, start, length))
            placed += 1
    return events


def inject_defects(generated, seed):
    """Return (corrupted copy, manifest). `generated` itself is never modified."""
    rng = np.random.default_rng(seed + cfg.DEFECT_SEED_OFFSET)
    df = generated.reset_index(drop=True).copy()
    df["_order"] = np.arange(len(df), dtype=float)
    position = {(int(e), int(m)): i for i, (e, m) in enumerate(zip(df.episode_id, df.minute))}

    manifest, extra_rows, to_drop = [], [], []
    for defect_id, (defect_type, episode, start, length) in enumerate(_choose_locations(df, rng)):
        i = position[(episode, start)]
        field, original, injected = "row", "", ""
        if defect_type == "missing_value":
            field = str(rng.choice(cfg.FEATURE_COLUMNS))
            original, injected = df.at[i, field], np.nan
            df.at[i, field] = injected
        elif defect_type == "negative_latency":
            field = "latency_ms"
            original = df.at[i, field]
            injected = df.at[i, field] = -original
        elif defect_type == "negative_throughput":
            field = "throughput_mbps"
            original = df.at[i, field]
            injected = df.at[i, field] = -original
        elif defect_type == "percent_over_100":
            field = "packet_loss_pct"
            original = df.at[i, field]
            injected = df.at[i, field] = original * 100
        elif defect_type == "throughput_over_capacity":
            field = "throughput_mbps"
            original = df.at[i, field]
            injected = df.at[i, field] = original + cfg.COUNTER_WRAP_MBPS
        elif defect_type == "invalid_category":
            field = cfg.STATUS_COLUMN
            original = df.at[i, field]
            injected = df.at[i, field] = str(rng.choice(["up", "UNKNOWN"]))
        elif defect_type == "inconsistent_status":
            field = cfg.STATUS_COLUMN
            original = df.at[i, field]
            injected = df.at[i, field] = "UP" if original == "UNSTABLE" else "UNSTABLE"
        elif defect_type == "frozen_value":
            field = "latency_ms,jitter_ms"
            source = df.loc[i, ["latency_ms", "jitter_ms"]].to_numpy()
            original = f"{_text(source[0])},{_text(source[1])}"
            injected = original
            for k in range(1, length):
                df.loc[i + k, ["latency_ms", "jitter_ms"]] = source
        elif defect_type == "plausible_offset":
            field = "latency_ms"
            original = df.at[i, field]
            injected = df.at[i, field] = original * cfg.PLAUSIBLE_OFFSET_FACTOR
        elif defect_type == "exact_duplicate":
            injected = "duplicate row"
            extra_rows.append(df.loc[[i]].assign(_order=df.at[i, "_order"] + 0.5))
        elif defect_type == "conflicting_duplicate":
            field = "latency_ms"
            original = df.at[i, field]
            injected = original * cfg.CONFLICT_FACTOR
            extra_rows.append(df.loc[[i]].assign(latency_ms=injected,
                                                 _order=df.at[i, "_order"] + 0.5))
        elif defect_type == "out_of_sequence":
            injected = "rows swapped"
            df.loc[[i, i + 1], "_order"] = df.loc[[i + 1, i], "_order"].to_numpy()
        elif defect_type == "timestamp_gap":
            injected = f"{length - 1} minutes removed"
            to_drop.extend(range(i, i + length - 1))

        expected, _ = DEFECT_TYPES[defect_type]
        manifest.append({"defect_id": defect_id, "defect_type": defect_type,
                         "episode_id": episode, "minute_start": start,
                         "minute_end": start + length - 1, "field": field,
                         "original_value": _text(original),
                         "injected_value": _text(injected),
                         "expected_check": expected, "detectable": expected != "none"})

    df = df.drop(index=to_drop)
    corrupted = (pd.concat([df, *extra_rows]).sort_values("_order", kind="stable")
                 .drop(columns="_order").reset_index(drop=True))
    return corrupted, pd.DataFrame(manifest, columns=MANIFEST_COLUMNS)


def footprint_mask(telemetry, manifest):
    """True for rows whose (episode, minute) lies inside any defect footprint."""
    touched = set()
    for event in manifest.itertuples():
        for minute in range(event.minute_start, event.minute_end + 1):
            touched.add((event.episode_id, minute))
    keys = zip(telemetry.episode_id.astype(int), telemetry.minute.astype(int))
    return pd.Series([k in touched for k in keys], index=telemetry.index)


def evaluate_against_manifest(df, flags, manifest):
    """Score the validation layer against the known injected defects.

    Returns (detection table per defect type, untouched-rows-flagged table per check).
    A detectable event counts as detected if its expected check flags at least one row
    in its footprint. For the undetectable `plausible_offset`, the table reports whether
    ANY FAIL/WARN check flagged it.
    """
    warn_or_fail = [c for c, (level, *_) in CHECKS.items() if level != INFO]
    detected = []
    for event in manifest.itertuples():
        in_footprint = ((df.episode_id == event.episode_id)
                        & df.minute.between(event.minute_start, event.minute_end))
        if event.detectable:
            detected.append(bool(flags.loc[in_footprint, event.expected_check].any()))
        else:
            detected.append(bool(flags.loc[in_footprint, warn_or_fail].any().any()))
    scored = manifest.assign(detected=detected)
    detection = (scored.groupby(["defect_type", "expected_check"], sort=False)
                 .agg(injected=("defect_id", "size"), detected=("detected", "sum"))
                 .reset_index())
    detection["missed"] = detection["injected"] - detection["detected"]
    detection.loc[detection.expected_check == "none", "note"] = \
        "undetectable by design; 'detected' = flagged by any FAIL/WARN check"

    untouched = ~footprint_mask(df, manifest)
    untouched_flags = pd.DataFrame({
        "check": list(CHECKS),
        "level": [CHECKS[c][0] for c in CHECKS],
        "untouched_rows_flagged": [int(flags.loc[untouched, c].astype(bool).sum()) for c in CHECKS],
    })
    return detection, untouched_flags
