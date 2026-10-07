"""Tests for deliberate defect injection and evaluation against the manifest."""

import numpy as np
import pandas as pd
import pytest

from netsense import config as cfg
from netsense.data_validation import CHECKS, FAIL, validate
from netsense.defect_injection import DEFECT_TYPES, evaluate_against_manifest, inject_defects


@pytest.fixture(scope="module")
def injected(generated):
    return inject_defects(generated, cfg.RANDOM_SEED)


@pytest.fixture(scope="module")
def evaluation(injected):
    corrupted, manifest = injected
    df, flags, _, _ = validate(corrupted)
    return evaluate_against_manifest(df, flags, manifest)


# 8. The generated input is never modified
def test_injection_leaves_its_input_unchanged(generated):
    before = generated.copy()
    inject_defects(generated, cfg.RANDOM_SEED)
    pd.testing.assert_frame_equal(generated, before)


# 9. Reproducibility
def test_injection_is_deterministic(generated, injected):
    corrupted, manifest = injected
    again_corrupted, again_manifest = inject_defects(generated, cfg.RANDOM_SEED)
    pd.testing.assert_frame_equal(corrupted, again_corrupted)
    pd.testing.assert_frame_equal(manifest, again_manifest)


# 10. The manifest describes what is really in the corrupted copy
def _rows(corrupted, event, minute=None):
    minute = event.minute_start if minute is None else minute
    return corrupted[(corrupted.episode_id == event.episode_id) & (corrupted.minute == minute)]


def test_manifest_matches_the_corrupted_copy(injected):
    corrupted, manifest = injected
    assert manifest.defect_type.value_counts().to_dict() == \
        {t: cfg.DEFECT_EVENTS_PER_TYPE for t in DEFECT_TYPES}
    for event in manifest.itertuples():
        rows = _rows(corrupted, event)
        kind = event.defect_type
        if kind == "missing_value":
            assert rows[event.field].isna().all()
        elif kind in ("invalid_category", "inconsistent_status"):
            assert (rows[event.field] == event.injected_value).all()
        elif kind in ("negative_latency", "negative_throughput", "percent_over_100",
                      "throughput_over_capacity", "plausible_offset"):
            assert rows[event.field].iloc[0] == pytest.approx(float(event.injected_value))
        elif kind == "conflicting_duplicate":
            assert len(rows) == 2
            assert rows.latency_ms.to_numpy() == pytest.approx(
                [float(event.original_value), float(event.injected_value)])
        elif kind == "exact_duplicate":
            assert len(rows) == 2 and rows.iloc[0].equals(rows.iloc[1])
        elif kind == "out_of_sequence":
            later = _rows(corrupted, event, event.minute_start + 1).index[0]
            assert later < rows.index[0]
        elif kind == "timestamp_gap":
            removed = range(event.minute_start, event.minute_end)
            assert all(_rows(corrupted, event, m).empty for m in removed)
        elif kind == "frozen_value":
            latency = float(event.injected_value.split(",")[0])
            for m in range(event.minute_start, event.minute_end + 1):
                assert _rows(corrupted, event, m).latency_ms.iloc[0] == latency


# 11. End-to-end evaluation, including the declared limit
def test_every_detectable_defect_is_detected(evaluation):
    detection, _ = evaluation
    detectable = detection[detection.expected_check != "none"]
    assert (detectable.detected == detectable.injected).all()


def test_plausible_offset_is_reported_as_missed(evaluation):
    """Declared limit: rule-based validation cannot prove plausible values are accurate."""
    detection, _ = evaluation
    row = detection.set_index("defect_type").loc["plausible_offset"]
    assert row.detected == 0 and row.missed == cfg.DEFECT_EVENTS_PER_TYPE


def test_fail_checks_flag_no_untouched_rows(evaluation):
    _, untouched = evaluation
    fail_rows = untouched[untouched.level == FAIL]
    assert (fail_rows.untouched_rows_flagged == 0).all()
