"""Tests for the Telemetry Integrity & Validation Layer.

Each test protects a declared behaviour: what each rule flags, what is kept, what is
quarantined, and that legitimate (including severe) telemetry is not rejected.
"""

import numpy as np
import pandas as pd
import pytest

from netsense import config as cfg
from netsense.data_validation import CHECKS, FAIL, run_checks, split_validated, validate

FAIL_CHECKS = [c for c, (level, *_) in CHECKS.items() if level == FAIL]


@pytest.fixture(scope="module")
def generated_validation(generated):
    return validate(generated)


@pytest.fixture
def stable_slice(generated):
    """Two stable episodes: no degradation, so no rule should fire before a mutation."""
    ids = generated.loc[generated.episode_type == cfg.STABLE, "episode_id"].unique()[:2]
    return generated[generated.episode_id.isin(ids)].reset_index(drop=True).copy()


def flagged_positions(df, check_id):
    _, flags = run_checks(df)
    return set(np.flatnonzero(flags[check_id].to_numpy(dtype=bool)))


# 1. Generated input: no FAIL; only O3 raises a warning
def test_generated_input_has_no_fail_and_only_o3_warns(generated_validation):
    _, flags, report, status = generated_validation
    assert not flags[FAIL_CHECKS].any().any()
    for check_id in ["S2-status", "S3-status", "C1", "O1", "O2"]:
        assert not flags[check_id].any(), check_id
    warned = report.loc[report.status == "warn", "check"].tolist()
    assert warned == ["O3"]
    assert status == "WARN"


# 2. Statistically unusual is not invalid
def test_unusual_values_are_reported_but_nothing_is_quarantined(generated_validation):
    df, flags, _, _ = generated_validation
    assert flags["U1"].sum() > 0
    validated, quarantined = split_validated(df, flags)
    assert len(validated) == len(df) and quarantined.empty


# 3. Each rule flags exactly the defect it is designed for
def _set(column, value, row=10):
    def mutate(df):
        df.loc[row, column] = value
        return df, {row}
    return mutate


def _duplicate_with_new_latency(df):
    extra = df.loc[[10]].assign(latency_ms=df.at[10, "latency_ms"] * 1.1)
    out = pd.concat([df.iloc[:11], extra, df.iloc[11:]]).reset_index(drop=True)
    return out, {10, 11}


def _repeat_timestamp(df):
    df.loc[11, "timestamp"] = df.at[10, "timestamp"]
    return df, {10, 11}


def _swap_rows(df):
    out = pd.concat([df.iloc[:10], df.iloc[[11, 10]], df.iloc[12:]]).reset_index(drop=True)
    return out, {11}


def _remove_rows(df):
    return df.drop(index=[10, 11, 12]).reset_index(drop=True), {10}


def _flip_status(df):
    df.loc[10, "connection_state"] = "UNSTABLE" if df.at[10, "connection_state"] == "UP" else "UP"
    return df, {10}


def _freeze_probe_metrics(df):
    df.loc[[11, 12], ["latency_ms", "jitter_ms"]] = df.loc[10, ["latency_ms", "jitter_ms"]].to_numpy()
    return df, {10, 11, 12}


def _freeze_throughput_only(df):
    df.loc[11:14, "throughput_mbps"] = df.at[10, "throughput_mbps"]
    return df, set()


@pytest.mark.parametrize("check_id, mutate", [
    ("S2-telemetry", _set("latency_ms", np.nan)),
    ("S2-status", _set("connection_state", None)),
    ("S2-labels", _set("state", None)),
    ("S3-status", _set("connection_state", "up")),
    ("S3-labels", _set("cause", "weather")),
    ("S4", _set("jitter_ms", -1.0)),
    ("S5", _set("error_rate_pct", 150.0)),
    ("S6", _set("throughput_mbps", cfg.LINK_CAPACITY_MBPS + 0.5)),
    ("I1", _duplicate_with_new_latency),
    ("I2", _repeat_timestamp),
    ("I3", _swap_rows),
    ("C1", _remove_rows),
    ("O1", _flip_status),
    ("O2", _freeze_probe_metrics),
    ("O2", _freeze_throughput_only),
    ("O3", _set("error_rate_pct", 100.0)),
])
def test_rule_flags_exactly_the_injected_row(stable_slice, check_id, mutate):
    assert flagged_positions(stable_slice, check_id) == set()
    mutated, expected = mutate(stable_slice)
    assert flagged_positions(mutated, check_id) == expected


def test_missing_status_is_warn_but_missing_telemetry_is_fail(stable_slice):
    stable_slice.loc[10, "connection_state"] = None
    stable_slice.loc[20, "throughput_mbps"] = np.nan
    df, flags = run_checks(stable_slice)
    validated, _ = split_validated(df, flags)
    assert 10 in validated.index and 20 not in validated.index


# 4. Boundary values are valid
def test_boundary_values_are_not_rejected(stable_slice):
    stable_slice.loc[10, "packet_loss_pct"] = 0.0
    stable_slice.loc[11, ["packet_loss_pct", "connection_state"]] = [100.0, "UNSTABLE"]
    stable_slice.loc[12, "retransmission_rate_pct"] = 100.0
    stable_slice.loc[13, "throughput_mbps"] = cfg.LINK_CAPACITY_MBPS
    stable_slice.loc[21, ["latency_ms", "jitter_ms"]] = \
        stable_slice.loc[20, ["latency_ms", "jitter_ms"]].to_numpy()  # 2 identical minutes
    _, flags = run_checks(stable_slice)
    assert not flags[FAIL_CHECKS + ["C1", "O1", "O2"]].any().any()


# 5. Schema problems fail gracefully
def test_missing_required_column_fails_without_crashing(stable_slice):
    _, _, report, status = validate(stable_slice.drop(columns="jitter_ms"))
    by_check = report.set_index("check")
    assert by_check.loc["S1-missing-required", "status"] == "fail"
    assert by_check.loc["S4", "status"] == "not evaluated"
    assert status == "FAIL"


def test_unexpected_column_is_a_warning(stable_slice):
    _, _, report, status = validate(stable_slice.assign(extra_field=1))
    assert report.set_index("check").loc["S1-unexpected", "status"] == "warn"
    assert status == "WARN"


# 6. Validation never depends on ground-truth label values
def test_flags_do_not_depend_on_ground_truth_values(generated):
    sample = generated.iloc[:5000].copy()
    _, original = run_checks(sample)
    altered = sample.assign(state=cfg.NORMAL, hidden_severity=0.0)
    _, changed = run_checks(altered)
    pd.testing.assert_frame_equal(original, changed)


# 7. Declared quarantine policy
def test_quarantine_policy(stable_slice):
    df = stable_slice
    exact = df.loc[[5]]
    conflict = df.loc[[20]].assign(latency_ms=df.at[20, "latency_ms"] * 1.1)
    df.loc[30, "latency_ms"] = np.nan              # missing telemetry -> quarantine
    df.loc[40, "connection_state"] = None          # missing status -> keep
    df.loc[50, "connection_state"] = "UNSTABLE"    # inconsistent status (WARN) -> keep
    df = df.drop(index=[60, 61, 62])               # gap -> not filled
    df = pd.concat([df, exact, conflict]).sort_index(kind="stable").reset_index(drop=True)

    prepared, flags = run_checks(df)
    validated, quarantined = split_validated(prepared, flags)
    first_episode = stable_slice.episode_id.iloc[0]
    minutes = validated[validated.episode_id == first_episode].minute

    assert (minutes == 5).sum() == 1                   # exact duplicate: one copy kept
    assert (minutes == 20).sum() == 0                  # conflicting duplicate: all removed
    assert (minutes == 30).sum() == 0                  # missing telemetry removed
    assert (minutes == 40).sum() == 1 and (minutes == 50).sum() == 1  # WARN rows kept
    assert not minutes.isin([60, 61, 62]).any()        # gap not filled
    assert len(validated) + len(quarantined) == len(prepared)
