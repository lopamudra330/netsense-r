"""Experiment 3: interpretable ML detector (logistic regression).

  python -m experiments.e3_ml_detector freeze
      Phase 1, training episodes only: build causal features, fit the scaler and model,
      derive tau, run ML-A/B/C1/C2 on training episodes, write the frozen specification,
      coefficient table, training results, per-minute diagnostics and Figure 8.
  python -m experiments.e3_ml_detector evaluate
      Phase 2 (held-out, authorised after commit 2de1733): reconstructs scores from the
      committed specification without refitting, and refuses to run if frozen files differ.

Scores are logistic model scores (predicted probability under the fitted logistic model);
probability calibration is not performed.
"""

import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from netsense import config as cfg
from netsense import evaluation as ev
from netsense import ml_detector as ml
from netsense.threshold_detector import suspicious_minutes
from experiments.e2_threshold_baseline import training_data

SPEC_PATH = cfg.METRICS_DIR / "e3_ml_spec.json"
E2_SPEC_PATH = cfg.METRICS_DIR / "e2_detector_spec.json"
VARIANTS = {"ML_" + name: kw for name, kw in cfg.DETECTORS.items()}
GROUP_LABELS = {"latency_ms": "Latency", "jitter_ms": "Jitter", "error_rate_pct": "Error rate",
                "retransmission_rate_pct": "Retransmission rate"}
KIND_LABELS = {"current": "current", "mean10": "10-min mean", "trend5": "5-min trend"}


def minute_diagnostics(rows, labels, scores, tau, ml_flag, eng_flag):
    """Per-minute diagnostics on fitting rows: ML scores, and ML vs engineering hit rates."""
    y = labels.to_numpy(int)
    table = {"metric": [], "ml": [], "engineering_condition": []}
    def add(name, ml_value, eng_value=np.nan):
        table["metric"].append(name); table["ml"].append(ml_value); table["engineering_condition"].append(eng_value)
    add("pr_auc", average_precision_score(y, scores))
    add("roc_auc", roc_auc_score(y, scores))
    add("precision_at_threshold", precision_score(y, ml_flag), precision_score(y, eng_flag))
    add("recall_at_threshold", recall_score(y, ml_flag), recall_score(y, eng_flag))
    add("f1_at_threshold", f1_score(y, ml_flag), f1_score(y, eng_flag))
    healthy = y == 0
    add("healthy_trigger_rate_pct", 100 * ml_flag[healthy].mean(), 100 * eng_flag[healthy].mean())
    for state in (cfg.EARLY_DEGRADATION, cfg.DEGRADED, cfg.SEVERE_DEGRADATION):
        mask = (rows["state"].to_numpy() == state) & (y == 1)
        add(f"hit_rate_{state}_pct", 100 * ml_flag[mask].mean(), 100 * eng_flag[mask].mean())
    return pd.DataFrame(table)


def figure_coefficients(coefs):
    import matplotlib.pyplot as plt
    from netsense import plotting as plot
    groups = list(GROUP_LABELS)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1.6, 1]})
    rows = coefs[coefs.feature.isin(ml.FEATURES)].set_index("feature").loc[ml.FEATURES]
    y_positions, labels, y = [], [], 0
    for g in groups:
        for kind in KIND_LABELS:
            y_positions.append(y); labels.append(f"{GROUP_LABELS[g]}: {KIND_LABELS[kind]}"); y += 1
        y += 0.8  # gap between indicator groups
    values = rows.standardised_coefficient.to_numpy()
    colours = ["#184f95" if v > 0 else "#b3302f" for v in values]  # diverging pair, not the cause colours
    left.barh(y_positions, values, color=colours, height=0.75)
    left.axvline(0, color=plot.INK, lw=0.8)
    left.set_yticks(y_positions, labels)
    left.invert_yaxis()
    left.grid(axis="x"); left.grid(axis="y", visible=False)
    for yy, v in zip(y_positions, values):
        left.text(v, yy, f" {v:+.2f} ", va="center", ha="left" if v > 0 else "right", fontsize=8)
    pad = max(abs(values)) * 0.25
    left.set_xlim(min(values.min(), 0) - pad, max(values.max(), 0) + pad)
    left.set_xlabel("Standardised coefficient (per 1 SD of the feature)")
    left.set_title("Model coefficients (descriptive, not causal)")

    sens = coefs.set_index("feature").loc[[f"drop_group:{g}" for g in groups], "pr_auc_change"].to_numpy()
    right.barh(range(len(groups)), sens, color=plot.INK_SECONDARY, height=0.6)
    right.set_yticks(range(len(groups)), [GROUP_LABELS[g] for g in groups])
    right.invert_yaxis()
    right.axvline(0, color=plot.INK, lw=0.8)
    right.grid(axis="x"); right.grid(axis="y", visible=False)
    for i, v in enumerate(sens):
        label = f" {v:+.1e} " if abs(v) < 1e-3 else f" {v:+.4f} "  # tiny values: no misleading "-0.0000"
        right.text(v, i, label, va="center", ha="left" if v > 0 else "right", fontsize=8)
    pad = max(abs(sens)) * 0.45
    right.set_xlim(min(sens.min(), 0) - pad, max(sens.max(), 0) + pad)
    right.set_xlabel("Change in training PR-AUC when the group is removed")
    right.set_title("Leave-one-indicator-group-out\nsensitivity diagnostic")
    fig.suptitle("Interpretable ML detector: what the logistic model uses (TRAINING episodes)", fontsize=12)
    fig.tight_layout()
    return plot.save(fig, "fig8_ml_coefficients.png",
                     "Figure 8. Logistic regression on 12 causal features, training episodes of five seeds. "
                     "Left: blue = positive, red = negative coefficient. Current, mean and trend features of one "
                     "indicator are correlated, so weight can redistribute among them. Right: refit without each "
                     "indicator's three features; a training sensitivity diagnostic, not independent importance.")


def freeze():
    began = time.perf_counter()
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    frames, episode_tables = [], []
    for seed in cfg.EXPERIMENT_SEEDS:  # training episodes only (Experiment 2's training path)
        telemetry, episodes, _ = training_data(seed)
        frames.append(telemetry); episode_tables.append(episodes)
    training = pd.concat(frames, ignore_index=True)
    episodes = pd.concat(episode_tables, ignore_index=True)
    truth = ev.ground_truth(training, episodes)

    features = ml.build_features(training)
    labels = ml.training_labels(training, truth)
    defined = features.notna().all(axis=1)
    fitting = defined & labels.notna()
    counts = {
        "training_episodes": int(truth.shape[0]), "training_rows": int(len(training)),
        "fitting_rows": int(fitting.sum()), "healthy_fitting_rows": int((fitting & (labels == 0)).sum()),
        "positive_fitting_rows": int((fitting & (labels == 1)).sum()),
        "excluded_warmup_rows": int((training.minute < cfg.ML_WINDOW_MIN - 1).sum()),
        "excluded_undefined_other_rows": int((~defined & (training.minute >= cfg.ML_WINDOW_MIN - 1)).sum()),
        "excluded_post_onset_low_severity_rows": int((defined & labels.isna()).sum()),
    }

    model_dict, model = ml.fit(features[fitting], labels[fitting])
    scores = ml.score(features, model_dict)
    reproduction_error = float(np.max(np.abs(
        scores[fitting].to_numpy() - model.predict_proba(
            (features[fitting].to_numpy() - np.array(model_dict["scaler_mean"])) / np.array(model_dict["scaler_std"]))[:, 1])))
    healthy_rows = fitting & (labels == 0)
    tau, achieved = ml.derive_threshold(scores[healthy_rows])
    ml_suspicious = (scores > tau).fillna(False).to_numpy()

    # Frozen engineering per-minute condition (Experiment 2 spec), on the same rows.
    e2_spec = json.loads(E2_SPEC_PATH.read_text())
    eng_suspicious = suspicious_minutes(training, e2_spec["per_minute_condition"]["thresholds"]).any(axis=1).to_numpy()
    diag = minute_diagnostics(training[fitting], labels[fitting], scores[fitting].to_numpy(),
                              tau, ml_suspicious[fitting.to_numpy()], eng_suspicious[fitting.to_numpy()])
    diag.to_csv(cfg.METRICS_DIR / "e3_minute_diagnostics_train.csv", index=False)

    # Coefficients and the leave-one-indicator-group-out sensitivity diagnostic (diagnostic only).
    base_pr_auc = average_precision_score(labels[fitting].astype(int), scores[fitting])
    coef_rows = [{"feature": f, "indicator": f.split("__")[0], "kind": f.split("__")[1],
                  "standardised_coefficient": c, "sign": "+" if c > 0 else "-",
                  "odds_ratio_per_sd": float(np.exp(c))}
                 for f, c in zip(model_dict["features"], model_dict["coefficients"])]
    for g in cfg.ML_INDICATORS:
        kept = [f for f in ml.FEATURES if not f.startswith(g + "__")]
        reduced, _ = ml.fit(features[fitting], labels[fitting], kept)
        pr = average_precision_score(labels[fitting].astype(int), ml.score(features[fitting], reduced))
        coef_rows.append({"feature": f"drop_group:{g}", "indicator": g, "kind": "sensitivity",
                          "pr_auc_without_group": pr, "pr_auc_change": pr - base_pr_auc})
    coefs = pd.DataFrame(coef_rows)
    coefs.to_csv(cfg.METRICS_DIR / "e3_coefficients.csv", index=False)

    # Event-level results with Experiment 2's frozen persistence/event/evaluation functions.
    results = []
    for name, (k, w) in VARIANTS.items():
        events = ev.classify_events(ml.ml_events(training, ml_suspicious, k, w, e2_spec["alarm_events"]["clear_minutes"]), truth)
        results.append({"detector": name, "scope": "pooled", **ev.summarise(events, truth),
                        **outcome_counts(events, truth)})
        for seed in cfg.EXPERIMENT_SEEDS:
            ce, ct = events[events.seed == seed], truth[truth.seed == seed]
            results.append({"detector": name, "scope": f"seed_{seed}", **ev.summarise(ce, ct), **outcome_counts(ce, ct)})
        for cause in cfg.CAUSES:
            results.append({"detector": name, "scope": f"cause_{cause}", **ev.summarise(events, truth, cause)})
    results = pd.DataFrame(results)
    results.to_csv(cfg.METRICS_DIR / "e3_results_train.csv", index=False)

    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    spec = {
        "description": "Frozen Experiment 3 interpretable ML detector specification (Phase 1).",
        "frozen_on": "2026-10-08",
        "score": "logistic model score (predicted probability under the fitted logistic model); "
                 "probability calibration is not performed",
        "inputs": {"indicators": cfg.ML_INDICATORS, "excluded": ["packet_loss_pct", "throughput_mbps",
                   "connection_state", "state", "hidden_severity", "cause", "episode_type", "episode_id",
                   "timestamp"]},
        "transform": {"log10_plus_offset": cfg.ML_LOG_INDICATORS, "offset": cfg.ML_LOG_OFFSET},
        "features": {"names": ml.FEATURES,
                     "current": "x(t)", "mean10": "mean of x(t-9..t)",
                     "trend5": "mean(x(t-4..t)) - mean(x(t-9..t-5))",
                     "scope": "within each (seed, episode_id) on the full minute index; causal",
                     "undefined": "if any of minutes t-9..t is missing for any indicator (gap or warm-up "
                                  "minutes 0-8), all features at t are undefined; such minutes are excluded "
                                  "from fitting and are not suspicious"},
        "target": {"1": "inside degrading phase with hidden state EARLY/DEGRADED/SEVERE",
                   "0": "healthy minute outside the degrading phase",
                   "excluded": "inside degrading phase with hidden state NORMAL (severity < 0.10)"},
        "model": {"type": "sklearn LogisticRegression, library defaults (L2, C=1.0, lbfgs)",
                  "max_iter": cfg.ML_MAX_ITER, "class_weight": None,
                  "standardisation": "population mean/std of training fitting rows", **model_dict},
        "threshold": {"tau": tau, "comparison": "suspicious if score > tau (strict)",
                      "target_healthy_rate": cfg.ML_TARGET_HEALTHY_RATE,
                      "achieved_healthy_rate": achieved,
                      "rule": "distinct healthy-training score s whose share of healthy scores strictly "
                              "above s is closest to the target; equally close -> larger s"},
        "persistence": {name: {"k": k, "w": w} for name, (k, w) in VARIANTS.items()},
        "alarm_events": "Experiment 2 frozen rule_satisfied/alarm_events/evaluation, clear_minutes = "
                        f"{e2_spec['alarm_events']['clear_minutes']}",
        "training": {"seeds": cfg.EXPERIMENT_SEEDS, **counts},
        "reproducibility": {"command": "python -m experiments.e3_ml_detector freeze",
                            "generator_sha256": sha(cfg.PROJECT_ROOT / "netsense" / "telemetry_generator.py"),
                            "e2_spec_sha256": sha(E2_SPEC_PATH),
                            "python": platform.python_version(), "numpy": np.__version__,
                            "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
    }
    SPEC_PATH.write_text(json.dumps(spec, indent=2) + "\n")

    from netsense import plotting as plot
    plot.apply_style()
    figure_coefficients(coefs)

    print(json.dumps(counts, indent=1))
    print(f"tau = {tau!r}; target healthy rate = {cfg.ML_TARGET_HEALTHY_RATE:.6f}; achieved = {achieved:.6f}")
    print(f"JSON reconstruction max abs score difference vs fitted model: {reproduction_error:.3e}")
    print(f"Runtime: {time.perf_counter() - began:.1f} s")


def outcome_counts(classified, truth):
    out = ev.episode_outcomes(classified, truth)
    return {"degrading_episodes": len(out),
            "valid_post_onset_n": int((out.outcome == ev.VALID).sum()),
            "alarm_active_at_onset_only_n": int((out.outcome == ev.ACTIVE_AT_ONSET).sum()),
            "no_valid_alarm_n": int((out.outcome == ev.NO_ALARM).sum())}


# --- Phase 2: single held-out evaluation of the frozen specification -------------------

FROZEN_COMMIT = "2de1733"
FROZEN_FILES = ["netsense/", "experiments/e2_threshold_baseline.py",
                "results/metrics/e3_ml_spec.json", "results/metrics/e3_coefficients.csv",
                "results/metrics/e3_results_train.csv", "results/metrics/e3_minute_diagnostics_train.csv",
                "results/figures/fig8_ml_coefficients.png", "results/metrics/e2_detector_spec.json",
                "results/metrics/e2_results_test.csv"]
PAIRS = {"ML_" + name: name for name in cfg.DETECTORS}
SHORT = {"A_single_minute": "A", "B_3_consecutive": "B", "C1_3_of_10": "C1", "C2_6_of_20": "C2"}
EXPECTED_TEST_STRATA = {(cfg.STABLE, cfg.NO_CAUSE): 40, (cfg.WORSENING, cfg.CONGESTION): 30,
                        (cfg.WORSENING, cfg.LINK_QUALITY): 30, (cfg.RECOVERING, cfg.CONGESTION): 15,
                        (cfg.RECOVERING, cfg.LINK_QUALITY): 15}
COMPARE = ["detection_rate_worsening_pct", "detection_rate_recovering_pct", "false_alarms_per_100h",
           "healthy_alarm_burden_pct", "delay_median", "warned_before_severe_pct", "severe_lead_median"]


def check_frozen_state():
    import subprocess
    if subprocess.run(["git", "diff", "--quiet", FROZEN_COMMIT, "--", *FROZEN_FILES],
                      cwd=cfg.PROJECT_ROOT).returncode != 0:
        raise SystemExit(f"Frozen files differ from {FROZEN_COMMIT}; Phase 2 refused.")


def run_phase2(telemetry, episodes, spec, e2_spec):
    """Score held-out data with the frozen JSON model and evaluate the four ML variants."""
    truth = ev.ground_truth(telemetry, episodes)
    features = ml.build_features(telemetry)
    scores = ml.score(features, spec["model"])                     # reconstruction, no fitting
    tau = spec["threshold"]["tau"]
    suspicious = (scores > tau).fillna(False).to_numpy()           # strict, as frozen

    labels = ml.training_labels(telemetry, truth)                  # evaluation-only target
    rows = features.notna().all(axis=1) & labels.notna()
    eng = suspicious_minutes(telemetry, e2_spec["per_minute_condition"]["thresholds"]).any(axis=1).to_numpy()
    diag = minute_diagnostics(telemetry[rows], labels[rows], scores[rows].to_numpy(), tau,
                              suspicious[rows.to_numpy()], eng[rows.to_numpy()])

    results, clear = [], e2_spec["alarm_events"]["clear_minutes"]
    for name, rule in spec["persistence"].items():
        events = ev.classify_events(ml.ml_events(telemetry, suspicious, rule["k"], rule["w"], clear), truth)
        results.append({"detector": name, "scope": "pooled", **ev.summarise(events, truth),
                        **outcome_counts(events, truth)})
        for seed in sorted(truth.seed.unique()):
            ce, ct = events[events.seed == seed], truth[truth.seed == seed]
            results.append({"detector": name, "scope": f"seed_{seed}", **ev.summarise(ce, ct),
                            **outcome_counts(ce, ct)})
        for cause in cfg.CAUSES:
            results.append({"detector": name, "scope": f"cause_{cause}", **ev.summarise(events, truth, cause)})
    counts = {"rows": int(len(telemetry)), "scored_rows": int(scores.notna().sum()),
              "diagnostic_rows": int(rows.sum())}
    return truth, pd.DataFrame(results), diag, counts


def _seed_range(df, detector, metric):
    s = df[(df.detector == detector) & df.scope.str.startswith("seed")][metric]
    return s.min(), s.max()


def judge(ml_res, eng_res):
    """Pre-declared matched-pair judgement on pooled held-out results (13.5/design section 11)."""
    rows = []
    for ml_name, eng_name in PAIRS.items():
        m = ml_res[(ml_res.detector == ml_name) & (ml_res.scope == "pooled")].iloc[0]
        e = eng_res[(eng_res.detector == eng_name) & (eng_res.scope == "pooled")].iloc[0]
        def dominates(a, b):
            return (((a.false_alarms_per_100h < b.false_alarms_per_100h and a.delay_median <= b.delay_median)
                     or (a.delay_median < b.delay_median and a.false_alarms_per_100h <= b.false_alarms_per_100h))
                    and a.detection_rate_worsening_pct >= b.detection_rate_worsening_pct)
        verdict = ("ML dominates under pre-declared rule" if dominates(m, e) else
                   "engineering dominates" if dominates(e, m) else "trade-off / neither dominates")
        row = {"pair": f"ML-{SHORT[eng_name]} vs Eng-{SHORT[eng_name]}", "verdict": verdict}
        for metric in COMPARE:
            row[f"ml_{metric}"], row[f"eng_{metric}"] = m[metric], e[metric]
            row[f"diff_{metric}"] = m[metric] - e[metric]
        for metric in ("false_alarms_per_100h", "delay_median"):  # descriptive seed-variability note
            spread = max(np.subtract(*_seed_range(ml_res, ml_name, metric)[::-1]),
                         np.subtract(*_seed_range(eng_res, eng_name, metric)[::-1]))
            row[f"{metric}_diff_within_seed_spread"] = bool(abs(m[metric] - e[metric]) < spread)
        rows.append(row)
    return pd.DataFrame(rows)


def train_vs_test(test_res):
    train = pd.read_csv(cfg.METRICS_DIR / "e3_results_train.csv")
    rows = []
    for name in PAIRS:
        tr = train[(train.detector == name) & (train.scope == "pooled")].iloc[0]
        te = test_res[(test_res.detector == name) & (test_res.scope == "pooled")].iloc[0]
        for metric in COMPARE:
            lo, hi = _seed_range(train, name, metric)
            rows.append({"detector": name, "metric": metric, "train": tr[metric], "test": te[metric],
                         "change": te[metric] - tr[metric], "train_seed_min": lo, "train_seed_max": hi,
                         "test_outside_train_seed_range": bool(te[metric] < lo or te[metric] > hi)})
    return pd.DataFrame(rows)


def figure_tradeoff(ml_res, eng_res):
    import matplotlib.pyplot as plt
    from netsense import plotting as plot
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.15, 1]})
    eng_colour, ml_colour = "#52514e", "#4a3aa7"
    for eng_name in cfg.DETECTORS:
        ml_name = "ML_" + eng_name
        pts = {}
        for name, res, colour, filled in ((eng_name, eng_res, eng_colour, False), (ml_name, ml_res, ml_colour, True)):
            p = res[(res.detector == name) & (res.scope == "pooled")].iloc[0]
            dlo, dhi = _seed_range(res, name, "delay_median")
            flo, fhi = _seed_range(res, name, "false_alarms_per_100h")
            x, y = p.delay_median, p.false_alarms_per_100h
            left.errorbar(x, y, xerr=[[x - dlo], [dhi - x]], yerr=[[y - flo], [fhi - y]], fmt="o", ms=8,
                          color=colour, mfc=colour if filled else "white", mew=1.6, capsize=3, elinewidth=1, zorder=3)
            pts[filled] = (x, y)
        left.plot([pts[False][0], pts[True][0]], [pts[False][1], pts[True][1]], color="#b5b3ad", lw=1, ls="--", zorder=1)
        for filled, (x, y) in pts.items():
            short = SHORT[eng_name]
            # layout only: move labels of points that sit close together
            offset = {(False, "B"): (7, -13), (False, "C1"): (-38, 7),
                      (True, "B"): (-36, -12), (True, "C1"): (-40, 6)}.get((filled, short), (7, 5))
            left.annotate(("ML-" if filled else "Eng-") + short, (x, y), xytext=offset,
                          textcoords="offset points", fontsize=8.5, color=ml_colour if filled else eng_colour)
    left.set_xlim(left=0); left.set_ylim(bottom=0)
    left.set_xlabel("Median detection delay after onset (minutes)")
    left.set_ylabel("False alarms per 100 healthy hours")
    left.set_title("False alarms vs delay (dashed: matched pair)")
    left.grid(axis="x")
    left.plot([], [], "o", color=eng_colour, mfc="white", mew=1.6, label="Engineering (Experiment 2)")
    left.plot([], [], "o", color=ml_colour, label="ML (logistic model)")
    left.legend(loc="upper left")

    names = [n for e in cfg.DETECTORS for n in (e, "ML_" + e)]
    labels = [("ML-" if n.startswith("ML_") else "Eng-") + SHORT[n.replace("ML_", "")] for n in names]
    x = np.arange(len(names)); width = 0.38
    for offset, kind, colour in ((-width / 2, "worsening", "#4a3aa7"), (width / 2, "recovering", "#1baf7a")):
        col = f"detection_rate_{kind}_pct"
        vals, lows, highs = [], [], []
        for n in names:
            res = ml_res if n.startswith("ML_") else eng_res
            v = res[(res.detector == n) & (res.scope == "pooled")][col].iloc[0]
            lo, hi = _seed_range(res, n, col)
            vals.append(v); lows.append(v - lo); highs.append(hi - v)
        bars = right.bar(x + offset, vals, width, color=colour, label=f"{kind.capitalize()} episodes",
                         yerr=[lows, highs], capsize=2, error_kw={"elinewidth": 0.8, "ecolor": "#52514e"})
        for bar, v in zip(bars, vals):
            right.text(bar.get_x() + bar.get_width() / 2, 3, f"{v:.0f}" if v == round(v) else f"{v:.1f}", ha="center", va="bottom",
                       fontsize=7, color="white", fontweight="bold", rotation=90)
    right.set_xticks(x, labels, rotation=45, ha="right")
    right.set_ylim(0, 110)
    right.set_ylabel("Episodes with a valid post-onset detection (%)")
    right.set_title("Detection rate by episode type (held-out test)")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.suptitle("Interpretable ML vs engineering baseline: HELD-OUT TEST episodes (five seeds)", fontsize=12, y=1.02)
    fig.tight_layout()
    return plot.save(fig, "fig7_ml_vs_engineering_tradeoff.png",
                     "Figure 7. Held-out test episodes, five seeds pooled; whiskers show the range across seeds. "
                     "Engineering values from the committed Experiment 2 results; ML from the frozen Experiment 3 "
                     "specification (no refitting). Same persistence rules and event definitions for both.")


def evaluate():
    from experiments.e2_threshold_baseline import test_data
    began = time.perf_counter()
    check_frozen_state()
    spec = json.loads(SPEC_PATH.read_text())
    e2_spec = json.loads(E2_SPEC_PATH.read_text())
    frames, episode_tables = [], []
    for seed in spec["training"]["seeds"]:
        telemetry, episodes, status = test_data(seed)   # asserts no training episode entered
        strata = telemetry.drop_duplicates("episode_id").groupby(["episode_type", "cause"]).size().to_dict()
        assert strata == EXPECTED_TEST_STRATA, (seed, strata)
        frames.append(telemetry); episode_tables.append(episodes)
    telemetry = pd.concat(frames, ignore_index=True)
    episodes = pd.concat(episode_tables, ignore_index=True)

    truth, results, diag, counts = run_phase2(telemetry, episodes, spec, e2_spec)
    eng = pd.read_csv(cfg.METRICS_DIR / "e2_results_test.csv")   # committed; not recomputed
    results.to_csv(cfg.METRICS_DIR / "e3_results_test.csv", index=False)
    diag.to_csv(cfg.METRICS_DIR / "e3_minute_diagnostics_test.csv", index=False)
    judge(results, eng).to_csv(cfg.METRICS_DIR / "e3_ml_vs_engineering.csv", index=False)
    train_vs_test(results).to_csv(cfg.METRICS_DIR / "e3_train_vs_test.csv", index=False)

    from netsense import plotting as plot
    plot.apply_style()
    figure_tradeoff(results, eng)
    print(f"Held-out episodes: {truth.shape[0]}; {counts}")
    print(f"Runtime: {time.perf_counter() - began:.1f} s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    {"freeze": freeze, "evaluate": evaluate}.get(mode, lambda: sys.exit(
        "usage: python -m experiments.e3_ml_detector [freeze|evaluate]"))()
