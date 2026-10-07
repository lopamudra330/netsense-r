"""Experiment 0: Telemetry Integrity & Validation Layer ("Trust Layer").

1. Validate the generated input (synthetically generated clean input).
2. Inject known defects into a COPY and record them in a manifest.
3. Validate the corrupted copy and score the layer against the manifest.
4. Outlier counterfactual: what a naive "delete statistical outliers" step would
   have removed, by network state.

Run from the repository root:  python -m experiments.e0_telemetry_validation
"""

import time

import pandas as pd

from netsense import config as cfg
from netsense.data_validation import split_validated, validate
from netsense.defect_injection import evaluate_against_manifest, inject_defects
from netsense.telemetry_generator import generate_dataset


def outlier_counterfactual(df, flags):
    """Share of minutes in each state that U1 marks as statistically unusual."""
    table = (df.assign(unusual=flags["U1"].astype(bool))
             .groupby("state")["unusual"].agg(minutes="size", flagged_unusual="sum")
             .reindex(cfg.STATES).reset_index())
    table["share_flagged_pct"] = (100 * table.flagged_unusual / table.minutes).round(1)
    table["share_of_all_unusual_pct"] = (
        100 * table.flagged_unusual / table.flagged_unusual.sum()).round(1)
    return table


def main(seed=cfg.RANDOM_SEED):
    began = time.perf_counter()
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 60)

    generated, _ = generate_dataset(seed)

    # 1. Generated input
    df, flags, report, status = validate(generated)
    validated, quarantined = split_validated(df, flags)
    report.to_csv(cfg.METRICS_DIR / f"validation_report_generated_seed{seed}.csv", index=False)
    print(f"Generated input: {len(df):,} rows -> overall status {status} "
          "(PASS would mean only that no defined rule found a problem)")
    print(f"Validated telemetry: {len(validated):,} rows kept, {len(quarantined):,} quarantined")
    print(report[["check", "level", "status", "affected", "episodes_affected"]].to_string(index=False))

    # 2-3. Corrupted copy, scored against the manifest
    corrupted, manifest = inject_defects(generated, seed)
    corrupted.to_csv(cfg.GENERATED_DATA_DIR / f"telemetry_corrupted_seed{seed}.csv", index=False)
    manifest.to_csv(cfg.METRICS_DIR / f"defect_manifest_seed{seed}.csv", index=False)
    cdf, cflags, creport, cstatus = validate(corrupted)
    creport.to_csv(cfg.METRICS_DIR / f"validation_report_corrupted_seed{seed}.csv", index=False)
    c_validated, c_quarantined = split_validated(cdf, cflags)
    detection, untouched = evaluate_against_manifest(cdf, cflags, manifest)
    detection.to_csv(cfg.METRICS_DIR / f"defect_detection_seed{seed}.csv", index=False)
    untouched.to_csv(cfg.METRICS_DIR / f"untouched_rows_flagged_seed{seed}.csv", index=False)
    print(f"\nCorrupted copy: {len(cdf):,} rows, {len(manifest)} injected defect events "
          f"-> overall status {cstatus}; {len(c_quarantined):,} rows quarantined")
    print(creport[["check", "level", "status", "affected"]].to_string(index=False))
    print("\nDetection against the manifest:")
    print(detection.to_string(index=False))
    print("\nFlags on untouched rows (outside every defect footprint):")
    print(untouched.to_string(index=False))

    # 4. Outlier counterfactual on the generated input
    counterfactual = outlier_counterfactual(df, flags)
    counterfactual.to_csv(cfg.METRICS_DIR / f"outlier_counterfactual_seed{seed}.csv", index=False)
    severe = df["state"] == cfg.SEVERE_DEGRADATION
    print("\nOutlier counterfactual (U1 is INFO only; nothing is removed):")
    print(counterfactual.to_string(index=False))
    print(f"Severe-degradation minutes preserved in validated telemetry: "
          f"{int(validated.state.eq(cfg.SEVERE_DEGRADATION).sum()):,} of {int(severe.sum()):,}")
    print(f"\nRuntime: {time.perf_counter() - began:.1f} s")


if __name__ == "__main__":
    main()
