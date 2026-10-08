"""Experiment 2: engineering monitoring baseline.

Two separate phases (docs/methodology.md, section 12.5):
  python -m experiments.e2_threshold_baseline freeze
      Training episodes only: derive thresholds, run the four detectors, write the frozen
      specification and training tables. Test episodes are dropped immediately after
      generation using episode metadata only; no test telemetry value is used.
  python -m experiments.e2_threshold_baseline evaluate
      Phase 2 (held-out test episodes, authorised after commit ddd57ca): loads the committed
      specification, never recomputes thresholds, and refuses to run if any frozen file
      differs from that commit.
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


# --- Phase 2: single held-out evaluation with the committed specification -----------------

FROZEN_COMMIT = "ddd57ca"
FROZEN_FILES = ["netsense/threshold_detector.py", "netsense/evaluation.py", "netsense/config.py",
                "netsense/telemetry_generator.py", "netsense/splits.py",
                "results/metrics/e2_detector_spec.json"]
SHORT = {"A_single_minute": "A (1 of 1)", "B_3_consecutive": "B (3 of 3)",
         "C1_3_of_10": "C1 (3 of 10)", "C2_6_of_20": "C2 (6 of 20)"}
RATE_COLOURS = {"worsening": "#4a3aa7", "recovering": "#1baf7a"}  # validated pair
FIG6_INDICATOR = {cfg.CONGESTION: "latency_ms", cfg.LINK_QUALITY: "error_rate_pct"}


def check_frozen_state():
    """Refuse to evaluate unless detector, evaluation, config, generator, split and spec are
    byte-identical to the committed pre-test specification."""
    import subprocess
    result = subprocess.run(["git", "diff", "--quiet", FROZEN_COMMIT, "--", *FROZEN_FILES],
                            cwd=cfg.PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Frozen files differ from {FROZEN_COMMIT}; Phase 2 refused.")


def test_data(seed):
    """Generate -> keep TEST episodes (metadata only) -> validate -> validated rows."""
    generated, episodes = generate_dataset(seed)
    split = split_episodes(generated[["episode_id", "episode_type", "cause"]], seed)
    test_ids = set(split.loc[split.split != TRAIN, "episode_id"])
    train_ids = set(split.loc[split.split == TRAIN, "episode_id"])
    test = generated[generated.episode_id.isin(test_ids)].reset_index(drop=True)
    test_episodes = episodes[episodes.episode_id.isin(test_ids)].reset_index(drop=True)
    del generated, episodes
    df, flags, _, status = validate(test)
    if status == "FAIL":
        raise RuntimeError(f"seed {seed}: test telemetry validation FAIL")
    validated, _ = split_validated(df, flags)
    assert not set(validated.episode_id) & train_ids, "a training episode entered Phase 2"
    return validated.assign(seed=seed), test_episodes.assign(seed=seed), status


def outcome_counts(classified, truth):
    out = ev.episode_outcomes(classified, truth)
    return {"degrading_episodes": len(out),
            "valid_post_onset_n": int((out.outcome == ev.VALID).sum()),
            "alarm_active_at_onset_only_n": int((out.outcome == ev.ACTIVE_AT_ONSET).sum()),
            "no_valid_alarm_n": int((out.outcome == ev.NO_ALARM).sum())}


def state_at_first_detection(classified, truth, telemetry):
    """Evaluation-only description: hidden state at each first valid detection."""
    out = ev.episode_outcomes(classified, truth)
    valid = out[out.outcome == ev.VALID]
    lookup = telemetry.set_index(EPISODE_KEY + ["minute"])[["state", "hidden_severity"]]
    keys = list(zip(valid.seed, valid.episode_id, valid.detection_minute.astype(int)))
    found = lookup.loc[keys].reset_index(drop=True)
    category = found.state.where(found.state != cfg.NORMAL, "after onset, severity < 0.10")
    return pd.DataFrame({"episode_type": valid.episode_type.to_numpy(), "category": category})


def run_phase2(telemetry, episodes, spec):
    """All Phase 2 computations for already-loaded held-out data and a loaded spec."""
    truth = ev.ground_truth(telemetry, episodes)
    thresholds = spec["per_minute_condition"]["thresholds"]
    clear = spec["alarm_events"]["clear_minutes"]
    results, by_cause, signature, states, classified_all = [], [], [], [], {}
    for name, d in spec["detectors"].items():
        classified = ev.classify_events(run_detector(telemetry, thresholds, d["k"], d["w"], clear), truth)
        classified_all[name] = classified
        results.append({"detector": name, "scope": "pooled", **ev.summarise(classified, truth),
                        **outcome_counts(classified, truth)})
        for seed in sorted(truth.seed.unique()):
            ce, ct = classified[classified.seed == seed], truth[truth.seed == seed]
            results.append({"detector": name, "scope": f"seed_{seed}", **ev.summarise(ce, ct),
                            **outcome_counts(ce, ct)})
        for cause in cfg.CAUSES:
            by_cause.append({"detector": name, "cause": cause, **ev.summarise(classified, truth, cause)})
        signature.append(ev.first_trigger_by_cause(classified, truth).assign(detector=name))
        states.append(state_at_first_detection(classified, truth, telemetry).assign(detector=name))
    return (truth, pd.DataFrame(results), pd.DataFrame(by_cause), pd.concat(signature),
            pd.concat(states), classified_all)


def train_vs_test(test_results):
    metrics = ["detection_rate_worsening_pct", "detection_rate_recovering_pct", "false_alarms_per_100h",
               "healthy_alarm_burden_pct", "delay_median", "warned_before_severe_pct", "severe_lead_median"]
    train = pd.read_csv(cfg.METRICS_DIR / "e2_results_train.csv").query("scope == 'pooled'")
    test = test_results.query("scope == 'pooled'")
    rows = []
    for name in test.detector:
        tr, te = train[train.detector == name].iloc[0], test[test.detector == name].iloc[0]
        for m in metrics:
            rows.append({"detector": name, "metric": m, "train": tr[m], "test": te[m],
                         "change": te[m] - tr[m]})
    return pd.DataFrame(rows)


def figure_tradeoff(results):
    import matplotlib.pyplot as plt
    from netsense import plotting as plot
    pooled = results[results.scope == "pooled"].set_index("detector")
    seeds = results[results.scope != "pooled"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    markers = dict(zip(SHORT, ["o", "s", "D", "^"]))
    for name in SHORT:
        p, s_ = pooled.loc[name], seeds[seeds.detector == name]
        x, y = p.delay_median, p.false_alarms_per_100h
        left.errorbar(x, y, xerr=[[x - s_.delay_median.min()], [s_.delay_median.max() - x]],
                      yerr=[[y - s_.false_alarms_per_100h.min()], [s_.false_alarms_per_100h.max() - y]],
                      fmt=markers[name], color=plot.INK, ms=8, capsize=3, elinewidth=1)
        offset = (8, -14) if name == "B_3_consecutive" else (8, 6)  # keep B's label off C1's whisker
        left.annotate(SHORT[name], (x, y), xytext=offset, textcoords="offset points", fontsize=9)
    left.set_xlabel("Median detection delay after onset (minutes)")
    left.set_ylabel("False alarms per 100 healthy hours")
    left.set_ylim(bottom=0)
    left.set_xlim(left=0)  # start at 0 so delay differences are not visually exaggerated
    left.set_title("Persistence trades false alarms for delay")
    left.grid(axis="x")

    x = np.arange(len(SHORT)); width = 0.36
    for offset, kind in zip((-width / 2, width / 2), ("worsening", "recovering")):
        col = f"detection_rate_{kind}_pct"
        values = [pooled.loc[n, col] for n in SHORT]
        lows = [v - seeds[seeds.detector == n][col].min() for v, n in zip(values, SHORT)]
        highs = [seeds[seeds.detector == n][col].max() - v for v, n in zip(values, SHORT)]
        bars = right.bar(x + offset, values, width, color=RATE_COLOURS[kind], label=f"{kind.capitalize()} episodes",
                         yerr=[lows, highs], capsize=3, error_kw={"elinewidth": 1, "ecolor": plot.INK_SECONDARY})
        for bar, v in zip(bars, values):
            right.text(bar.get_x() + bar.get_width() / 2, 4, f"{v:.0f}%", ha="center", va="bottom",
                       fontsize=8, color="white", fontweight="bold")
    right.set_xticks(x, [SHORT[n] for n in SHORT])
    right.set_ylim(0, 110)
    right.set_ylabel("Episodes with a valid post-onset detection (%)")
    right.set_title("Detection rate by episode type")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.suptitle("Engineering baseline on HELD-OUT TEST episodes (five seeds)", fontsize=12, y=1.02)
    fig.tight_layout()
    return plot.save(fig, "fig5_detector_tradeoff.png",
                     "Figure 5. Held-out test episodes, five seeds pooled (points and bars); whiskers show "
                     "the range across the five seeds. Frozen Phase 1 detector specification.")


def select_example_episodes(truth):
    """Pre-declared example-episode rule, stated operationally.

    For each cause, select the seed-42 held-out worsening episode whose onset-to-severe
    duration is closest to the sample median onset-to-severe duration; if several episodes
    are equally close, choose the lowest episode_id. With an even number of episodes the
    sample median may lie between observed durations, so no episode need equal it exactly.
    """
    chosen = {}
    candidates = truth[(truth.seed == cfg.RANDOM_SEED) & (truth.episode_type == cfg.WORSENING)]
    for cause in cfg.CAUSES:
        c = candidates[candidates.cause == cause].assign(
            duration=lambda t: t.severe_entry - t.onset_minute)
        median = c.duration.median()
        c = c.assign(distance=(c.duration - median).abs()).sort_values(["distance", "episode_id"])
        row = c.iloc[0]
        chosen[cause] = {"seed": int(row.seed), "episode_id": int(row.episode_id),
                         "duration": float(row.duration), "median_duration": float(median),
                         "candidates": len(c)}
    return chosen


def figure_example(telemetry, truth, classified_all, thresholds, chosen):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from netsense import plotting as plot
    shade = {cfg.EARLY_DEGRADATION: "#efeeea", cfg.DEGRADED: "#e1dfd9", cfg.SEVERE_DEGRADATION: "#cfccc4"}
    fig = plt.figure(figsize=(11, 8.4))
    fig.set_size_inches(11, 9)
    grid = fig.add_gridspec(5, 1, height_ratios=[3, 1.2, 0.35, 3, 1.2], hspace=0.3)
    for row, cause in enumerate(cfg.CAUSES):
        pick = chosen[cause]
        key = (pick["seed"], pick["episode_id"])
        ep = telemetry[(telemetry.seed == key[0]) & (telemetry.episode_id == key[1])].sort_values("minute")
        t = truth[(truth.seed == key[0]) & (truth.episode_id == key[1])].iloc[0]
        indicator = FIG6_INDICATOR[cause]
        top = fig.add_subplot(grid[3 * row])
        lane = fig.add_subplot(grid[3 * row + 1], sharex=top)
        for state, colour in shade.items():
            minutes = ep.minute[ep.state == state]
            for start, stop in _spans(minutes.to_numpy()):
                top.axvspan(start - 0.5, stop + 0.5, color=colour, lw=0, zorder=0)
        top.plot(ep.minute, ep[indicator], color=plot.CAUSE_COLOURS[cause], lw=1.2, zorder=3)
        top.axhline(thresholds[indicator], color=plot.INK, ls="--", lw=1, zorder=4)
        top.text(ep.minute.max(), thresholds[indicator], " frozen threshold", va="bottom", ha="right",
                 fontsize=8, color=plot.INK)
        for x_, label in ((t.onset_minute, "onset"), (t.severe_entry, "severe entry")):
            top.axvline(x_, color=plot.INK_SECONDARY, ls=":", lw=1, zorder=4)
            top.text(x_, 1.0, f" {label}", transform=top.get_xaxis_transform(), va="top",
                     fontsize=8, color=plot.INK_SECONDARY)
        if indicator == "error_rate_pct":
            top.set_yscale("log")
        top.set_ylabel(plot.METRIC_LABELS[indicator].replace("Frame error rate", "Error rate"))
        top.set_title(f"{plot.CAUSE_LABELS[cause]}: test episode (seed {key[0]}, episode {key[1]}), "
                      f"onset→severe {pick['duration']:.0f} min (median {pick['median_duration']:.1f})",
                      fontsize=10, loc="left")
        plt.setp(top.get_xticklabels(), visible=False)
        names = list(SHORT)
        for i, name in enumerate(names):
            evs = classified_all[name]
            evs = evs[(evs.seed == key[0]) & (evs.episode_id == key[1])]
            y = len(names) - 1 - i
            for e in evs.itertuples():
                colour = plot.INK if e.kind == "valid_detection" else "#b5b3ad"
                lane.plot([e.start, e.online_end], [y, y], color=colour, lw=5, solid_capstyle="butt")
            valid = evs[evs.kind == "valid_detection"]
            if len(valid):
                first = valid.start.min()
                lane.plot(first, y, marker="v", color=plot.INK, ms=7, zorder=5)
                lane.annotate(f"{first:.0f}", (first, y), xytext=(-4, 0), textcoords="offset points",
                              ha="right", va="center", fontsize=7.5, color=plot.INK_SECONDARY)
        lane.axvline(t.onset_minute, color=plot.INK_SECONDARY, ls=":", lw=1)
        lane.axvline(t.severe_entry, color=plot.INK_SECONDARY, ls=":", lw=1)
        lane.set_yticks(range(len(names)), [SHORT[n].split()[0] for n in reversed(names)])
        lane.set_ylim(-0.7, len(names) - 0.3)
        lane.grid(False)
        lane.set_xlabel("Minute of episode" if row == 1 else "")
        lane.set_ylabel("Alarms", fontsize=8)
    handles = [Patch(color=c, label=plot.STATE_LABELS[s]) for s, c in shade.items()]
    handles.append(Patch(facecolor="none", edgecolor="none", label="(hidden state: evaluation only)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.95), fontsize=8.5)
    fig.suptitle("How the four frozen detectors respond to one held-out degrading link", fontsize=12, y=0.99)
    return plot.save(fig, "fig6_example_alarm_timeline.png",
                     "Figure 6. Episodes chosen by the pre-declared rule (seed 42, held-out worsening episode "
                     "with the median onset-to-severe duration per cause). Bars: online alarm spans (black: valid "
                     "detection; grey: false alarm). Triangle and number: first valid post-onset detection. Shading "
                     "shows the hidden state for explanation only; detectors never see it.")


def _spans(minutes):
    """Consecutive runs in a sorted integer array, as (start, stop) pairs."""
    if len(minutes) == 0:
        return []
    breaks = np.flatnonzero(np.diff(minutes) != 1)
    starts = np.r_[minutes[0], minutes[breaks + 1]]
    stops = np.r_[minutes[breaks], minutes[-1]]
    return list(zip(starts, stops))


def evaluate():
    began = time.perf_counter()
    check_frozen_state()
    spec = json.loads(SPEC_PATH.read_text())
    frames, episode_tables, statuses = [], [], {}
    for seed in spec["training"]["seeds"]:
        telemetry, episodes, status = test_data(seed)
        frames.append(telemetry); episode_tables.append(episodes); statuses[seed] = status
    telemetry = pd.concat(frames, ignore_index=True)
    episodes = pd.concat(episode_tables, ignore_index=True)

    truth, results, by_cause, signature, states, classified_all = run_phase2(telemetry, episodes, spec)
    results.to_csv(cfg.METRICS_DIR / "e2_results_test.csv", index=False)
    by_cause.to_csv(cfg.METRICS_DIR / "e2_results_test_by_cause.csv", index=False)
    signature.to_csv(cfg.METRICS_DIR / "e2_first_indicator_by_cause.csv", index=False)
    comparison = train_vs_test(results)
    comparison.to_csv(cfg.METRICS_DIR / "e2_train_vs_test.csv", index=False)

    from netsense import plotting as plot
    plot.apply_style()
    chosen = select_example_episodes(truth)
    figure_tradeoff(results)
    figure_example(telemetry, truth, classified_all, spec["per_minute_condition"]["thresholds"], chosen)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    print("Validation status (test rows):", statuses)
    print(f"Held-out episodes: {truth.shape[0]} {truth.episode_type.value_counts().to_dict()}")
    print("State at first valid detection (evaluation-only):")
    print(states.groupby(["detector", "episode_type"]).category.value_counts().unstack(fill_value=0).to_string())
    print("Example episodes:", chosen)
    print(f"Runtime: {time.perf_counter() - began:.1f} s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"freeze": freeze, "evaluate": evaluate}.get(mode, lambda: sys.exit(
        "usage: python -m experiments.e2_threshold_baseline [freeze|evaluate]"))()
