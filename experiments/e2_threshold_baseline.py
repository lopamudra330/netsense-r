"""Experiment 2: engineering monitoring baseline.

Two separate phases (docs/methodology.md, section 12.5):
  python -m experiments.e2_threshold_baseline freeze
      Training episodes only: derive thresholds, run the four detectors, write the frozen
      specification and training tables. Test episodes are dropped immediately after
      generation using episode metadata only; no test telemetry value is used.
  python -m experiments.e2_threshold_baseline evaluate
      Phase 2 (held-out test episodes). Not yet authorised; this mode only refuses to run.
"""

import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd

from netsense import config as cfg
from netsense import evaluation as ev
from netsense.data_validation import split_validated, validate
from netsense.splits import TRAIN, split_episodes
from netsense.telemetry_generator import generate_dataset
from netsense.threshold_detector import EPISODE_KEY, derive_thresholds, run_detector, suspicious_minutes

SPEC_PATH = cfg.METRICS_DIR / "e2_detector_spec.json"


def select_training(generated, episodes, seed):
    """Keep only training episodes, decided from episode metadata alone (id, type, cause).

    Returns (training telemetry, training episode table). Test rows are discarded here,
    before any telemetry value is validated, summarised or used.
    """
    split = split_episodes(generated[["episode_id", "episode_type", "cause"]], seed)
    train_ids = set(split.loc[split.split == TRAIN, "episode_id"])
    return (generated[generated.episode_id.isin(train_ids)].reset_index(drop=True),
            episodes[episodes.episode_id.isin(train_ids)].reset_index(drop=True))


def training_data(seed):
    """Generate -> keep training episodes -> validate (stop on FAIL) -> validated rows."""
    generated, episodes = generate_dataset(seed)
    training, train_episodes = select_training(generated, episodes, seed)
    del generated, episodes  # nothing below sees test episodes
    df, flags, _, status = validate(training)
    if status == "FAIL":
        raise RuntimeError(f"seed {seed}: training telemetry validation FAIL")
    validated, _ = split_validated(df, flags)
    return validated.assign(seed=seed), train_episodes.assign(seed=seed), status


def healthy_mask(telemetry, truth):
    """True for rows outside their episode's degrading phase (evaluation-only ground truth)."""
    phase = telemetry[EPISODE_KEY].merge(truth[EPISODE_KEY + ["onset_minute", "degrading_end"]],
                                         on=EPISODE_KEY, how="left")
    minute = telemetry.minute.to_numpy()
    inside = (minute >= phase.onset_minute.to_numpy()) & (minute <= phase.degrading_end.to_numpy())
    return pd.Series(~inside, index=telemetry.index)


def healthy_exceedance_runs(telemetry, truth, thresholds):
    """Lengths of consecutive suspicious-minute runs during healthy time (descriptive)."""
    healthy = healthy_mask(telemetry, truth)
    t = telemetry.assign(suspicious=suspicious_minutes(telemetry, thresholds).any(axis=1) & healthy)
    t = t.sort_values(EPISODE_KEY + ["minute"])
    runs = []
    for _, ep in t.groupby(EPISODE_KEY):
        values = ep.suspicious.to_numpy()
        edges = np.flatnonzero(np.diff(np.r_[0, values.astype(int), 0]))
        runs.extend(edges[1::2] - edges[::2])
    runs = pd.Series(runs, dtype=int)
    bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 10_000)]
    labels = ["1", "2", "3", "4-5", "6-10", ">10"]
    table = pd.DataFrame({"run_length_min": labels,
                          "runs": [int(runs.between(lo, hi).sum()) for lo, hi in bins]})
    table["share_pct"] = 100 * table.runs / max(len(runs), 1)
    return table, int(healthy.sum()), int(t.suspicious.sum())


def freeze():
    began = time.perf_counter()
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    frames, episode_tables, statuses = [], [], {}
    for seed in cfg.EXPERIMENT_SEEDS:
        telemetry, episodes, status = training_data(seed)
        frames.append(telemetry); episode_tables.append(episodes); statuses[seed] = status
    training = pd.concat(frames, ignore_index=True)
    episodes = pd.concat(episode_tables, ignore_index=True)
    truth = ev.ground_truth(training, episodes)

    # Thresholds: pooled over all seeds (used) and per seed (stability only, never used).
    thresholds, normal_rows = derive_thresholds(training)
    per_seed = {s: derive_thresholds(f)[0] for s, f in zip(cfg.EXPERIMENT_SEEDS, frames)}
    threshold_table = pd.DataFrame(
        [{"indicator": c, "frozen_threshold": thresholds[c],
          "per_seed_min": min(v[c] for v in per_seed.values()),
          "per_seed_max": max(v[c] for v in per_seed.values()),
          **{f"seed_{s}": per_seed[s][c] for s in cfg.EXPERIMENT_SEEDS}}
         for c in cfg.DETECTOR_INDICATORS])
    threshold_table.to_csv(cfg.METRICS_DIR / "e2_thresholds.csv", index=False)

    # Healthy behaviour of the per-minute condition.
    runs, healthy_minutes, suspicious_healthy = healthy_exceedance_runs(training, truth, thresholds)
    runs.to_csv(cfg.METRICS_DIR / "e2_healthy_exceedance_runs.csv", index=False)

    # Detectors on training episodes.
    results, by_cause, signature = [], [], []
    for name, (k, w) in cfg.DETECTORS.items():
        events = ev.classify_events(run_detector(training, thresholds, k, w), truth)
        results.append({"detector": name, "scope": "pooled", **ev.summarise(events, truth)})
        for seed in cfg.EXPERIMENT_SEEDS:
            mask_e, mask_t = events.seed == seed, truth.seed == seed
            results.append({"detector": name, "scope": f"seed_{seed}",
                            **ev.summarise(events[mask_e], truth[mask_t])})
        for cause in cfg.CAUSES:
            by_cause.append({"detector": name, "cause": cause, **ev.summarise(events, truth, cause)})
        signature.append(ev.first_trigger_by_cause(events, truth).assign(detector=name))
    results = pd.DataFrame(results)
    results.to_csv(cfg.METRICS_DIR / "e2_results_train.csv", index=False)
    pd.DataFrame(by_cause).to_csv(cfg.METRICS_DIR / "e2_results_train_by_cause.csv", index=False)
    pd.concat(signature).to_csv(cfg.METRICS_DIR / "e2_first_trigger_by_cause_train.csv", index=False)

    generator_sha = hashlib.sha256((cfg.PROJECT_ROOT / "netsense" / "telemetry_generator.py")
                                   .read_bytes()).hexdigest()
    spec = {
        "description": "Frozen Experiment 2 engineering detector specification (Phase 1). "
                       "Everything needed to apply the detector without recomputing anything.",
        "frozen_on": "2026-10-08",
        "inputs": {
            "indicators": cfg.DETECTOR_INDICATORS,
            "observation_order": "minutes ordered by (seed, episode_id, minute); each episode independent",
            "excluded_from_detection": ["packet_loss_pct", "throughput_mbps", "connection_state",
                                        "state", "hidden_severity", "cause", "episode_type"],
            "missing_values": "a missing value or missing minute counts as not suspicious",
        },
        "per_minute_condition": {
            "combination": "OR: a minute is suspicious if ANY indicator exceeds its threshold",
            "comparison": "strict: value > threshold (a value equal to the threshold is not suspicious)",
            "thresholds": thresholds,
            "threshold_rule": f"{cfg.DETECTOR_THRESHOLD_PERCENTILE}th percentile of NORMAL minutes in the "
                              "training episodes of all seeds pooled; a pre-declared empirical engineering "
                              "threshold from healthy training telemetry, not a telecommunications alarm standard",
        },
        "persistence_rule": "rule satisfied at minute t if at least k of the minutes t-w+1..t are suspicious "
                            "(fewer than w minutes are available at the start of an episode); counters reset "
                            "at the start of every episode",
        "detectors": {name: {"k": k, "w": w} for name, (k, w) in cfg.DETECTORS.items()},
        "alarm_events": {
            "start": "first minute the rule is satisfied while no alarm is active",
            "continuation": "the event continues while the rule is satisfied, and also if the rule is "
                            "satisfied again before closure is confirmed",
            "clear_minutes": cfg.ALARM_CLEAR_MINUTES,
            "closure_confirmation": "causal: confirmed on the clear_minutes-th consecutive minute in which "
                                    "the rule is not satisfied; no future observation is ever used",
            "effective_end": "last minute in which the rule was satisfied (retrospective description)",
            "online_span": "from start to closure confirmation (or episode end if never confirmed)",
            "episode_end": "an event still open at the end of an episode is closed there, unconfirmed",
            "counting": "one continuous alarm is one event; an episode may contain several events",
        },
        "training": {
            "seeds": cfg.EXPERIMENT_SEEDS,
            "split": f"frozen episode-level split (netsense/splits.py), seed + {cfg.SPLIT_SEED_OFFSET}",
            "episodes": int(truth.shape[0]),
            "normal_rows_for_thresholds": normal_rows,
        },
        "reproducibility": {
            "command": "python -m experiments.e2_threshold_baseline freeze",
            "generator_sha256": generator_sha,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n")

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("Validation status (training rows):", statuses)
    print(f"Training episodes: {truth.shape[0]} (per type: "
          f"{truth.episode_type.value_counts().to_dict()}); NORMAL rows for thresholds: {normal_rows:,}")
    print(threshold_table.round(4).to_string(index=False))
    print(f"\nHealthy minutes: {healthy_minutes:,}; suspicious: {suspicious_healthy:,} "
          f"({100 * suspicious_healthy / healthy_minutes:.2f}%)")
    healthy_rows = training[healthy_mask(training, truth)]
    per_indicator = {c: round(100 * (healthy_rows[c] > thresholds[c]).mean(), 2)
                     for c in cfg.DETECTOR_INDICATORS}
    print("Healthy suspicious rate per indicator (%):", per_indicator)
    print(runs.round(1).to_string(index=False))
    print(f"\nRuntime: {time.perf_counter() - began:.1f} s; spec written to {SPEC_PATH.name}")


def evaluate():
    raise SystemExit("Phase 2 (held-out evaluation) is not authorised yet. "
                     "It will load the committed specification and never recompute thresholds.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"freeze": freeze, "evaluate": evaluate}.get(mode, lambda: sys.exit(
        "usage: python -m experiments.e2_threshold_baseline [freeze|evaluate]"))()
