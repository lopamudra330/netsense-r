"""Experiment 1: network behaviour analysis (training episodes only).

What does degradation look like in telemetry, and does it differ by cause?
1. Distributions and summary statistics by state and cause.
2. Separability of EARLY/DEGRADED from NORMAL: pooled and per-link baseline-relative.
3. Spearman correlation between metrics, per cause.
4. Rising phase: do recovering and worsening episodes look different before they diverge?

Episodes are identified by (seed, episode_id). No test episode is used.
Run from the repository root:  python -m experiments.e1_network_behaviour
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd

from netsense import config as cfg
from netsense import plotting as plot
from netsense.behaviour_stats import (EPISODE_KEY, METRICS, baseline_relative,
                                      probability_of_superiority, rising_summaries,
                                      rising_window, separability, summary_by_state_cause)
from netsense.data_validation import split_validated, validate
from netsense.splits import TRAIN, split_episodes
from netsense.telemetry_generator import generate_dataset

FIG4_METRICS = {cfg.CONGESTION: ["latency_ms", "jitter_ms"],
                cfg.LINK_QUALITY: ["error_rate_pct", "retransmission_rate_pct"]}


def validated_training_telemetry(seed):
    """Generate -> validate (stop on FAIL) -> keep validated rows of training episodes."""
    generated, _ = generate_dataset(seed)
    df, flags, _, status = validate(generated)
    if status == "FAIL":
        raise RuntimeError(f"seed {seed}: validation status FAIL; Experiment 1 will not run")
    validated, _ = split_validated(df, flags)
    split = split_episodes(validated, seed)
    train_ids = set(split.loc[split.split == TRAIN, "episode_id"])
    training = validated[validated.episode_id.isin(train_ids)].assign(seed=seed)
    test_ids = set(split.loc[split.split != TRAIN, "episode_id"])
    assert not set(training.episode_id) & test_ids, "a test episode entered the analysis"
    return training, status, split


# --- Figures --------------------------------------------------------------------

def figure_telemetry_by_state(training):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    width = 0.34
    for ax, metric in zip(axes.flat, METRICS):
        for offset, cause in zip((-0.19, 0.19), cfg.CAUSES):
            rows = training[training.cause == cause]
            data = [rows.loc[rows.state == s, metric].dropna() for s in cfg.STATES]
            parts = ax.boxplot(data, positions=np.arange(4) + offset, widths=width,
                               whis=(5, 95), showfliers=False, patch_artist=True,
                               medianprops={"color": plot.INK, "linewidth": 1.2},
                               whiskerprops={"color": plot.INK_SECONDARY, "linewidth": 0.8},
                               capprops={"color": plot.INK_SECONDARY, "linewidth": 0.8},
                               boxprops={"linewidth": 0})
            for box in parts["boxes"]:
                box.set_facecolor(plot.CAUSE_COLOURS[cause])
        if metric in ("error_rate_pct", "retransmission_rate_pct"):
            ax.set_yscale("log")
        ax.set_xticks(range(4), [plot.STATE_LABELS[s] for s in cfg.STATES])
        ax.set_title(plot.METRIC_LABELS[metric] + (" — log scale" if ax.get_yscale() == "log" else ""))
    handles = [plt.Rectangle((0, 0), 1, 1, color=plot.CAUSE_COLOURS[c]) for c in cfg.CAUSES]
    fig.legend(handles, [plot.CAUSE_LABELS[c] for c in cfg.CAUSES], loc="upper center",
               ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("How telemetry changes as a link degrades, by cause", y=1.07, fontsize=12)
    fig.tight_layout()
    return plot.save(fig, "fig2_telemetry_by_state.png",
                     "Figure 2. Training episodes, seed 42. Boxes: middle 50% of minutes; line: "
                     "median; whiskers: 5th–95th percentile.")


def figure_correlation(correlations):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    labels = [plot.SHORT_METRIC_LABELS[m] for m in METRICS]
    for ax, cause in zip(axes, cfg.CAUSES):
        matrix = correlations[cause]
        image = ax.imshow(matrix.to_numpy(), cmap=plot.DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(6), labels, rotation=35, ha="right")
        ax.set_yticks(range(6), labels if cause == cfg.CAUSES[0] else [])  # same order both panels
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(6):
            for j in range(6):
                value = matrix.iat[i, j]
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) > 0.6 else plot.INK)
        ax.set_title(plot.CAUSE_LABELS[cause])
    colourbar = fig.colorbar(image, ax=axes, shrink=0.85)
    colourbar.set_label("Spearman correlation")
    fig.suptitle("Which telemetry variables carry overlapping information?", fontsize=12)
    return plot.save(fig, "fig3_correlation_by_cause.png",
                     "Figure 3. Spearman (rank) correlation over all minutes of training episodes "
                     "of each cause, seed 42. Descriptive only: not causal.")


def figure_rising_phase(summaries, superiority):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.8))
    rng = np.random.default_rng(0)  # horizontal jitter of dots only
    panels = [(c, m) for c in cfg.CAUSES for m in FIG4_METRICS[c]]
    for ax, (cause, metric) in zip(axes, panels):
        rows = summaries[summaries.cause == cause]
        for x, episode_type in enumerate((cfg.RECOVERING, cfg.WORSENING)):
            values = rows.loc[rows.episode_type == episode_type, f"level_{metric}"].dropna()
            jitter = rng.uniform(-0.18, 0.18, len(values))
            filled = episode_type == cfg.WORSENING
            ax.scatter(x + jitter, values, s=12, linewidths=0.8, alpha=0.75,
                       facecolors=plot.CAUSE_COLOURS[cause] if filled else "none",
                       edgecolors=plot.CAUSE_COLOURS[cause])
            ax.hlines(values.median(), x - 0.3, x + 0.3, color=plot.INK, linewidth=1.6)
        p = superiority.query("cause == @cause and metric == @metric and measure == 'level'")
        ax.set_yscale("log")  # ratios; a few extreme episodes would otherwise flatten the rest
        ax.yaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))  # label 1, 2, 5, 10, 20, ...
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # Readability only: if the highest point sits just above the last labelled tick,
        # extend the axis to the next 1-2-5 tick so its magnitude can be read.
        highest = rows[f"level_{metric}"].max()
        next_tick = min(t * 10.0 ** k for k in range(-1, 4) for t in (1, 2, 5) if t * 10.0 ** k >= highest)
        if next_tick <= 1.5 * highest:
            ax.set_ylim(top=next_tick * 1.08)
        ax.set_xticks([0, 1], ["Recovering", "Worsening"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{plot.CAUSE_LABELS[cause].split()[0]}: {plot.SHORT_METRIC_LABELS[metric]}"
                     f"\nP(worsening > recovering) = {p.p_superiority_pooled.iloc[0]:.2f}",
                     fontsize=9)
        ax.set_ylabel("× own baseline (log scale)" if ax is axes[0] else "")
    fig.suptitle("Before trajectories diverge, do recovering and worsening episodes look different?",
                 fontsize=12, y=1.04)
    fig.tight_layout()
    return plot.save(fig, "fig4_rising_phase_comparison.png",
                     "Figure 4. One dot per training episode, five seeds pooled. Value at the end of "
                     "the shared rising window (severity reaching 0.25), relative to the episode's "
                     "own first hour. Bar: median. 0.50 = no difference.")


# --- Main --------------------------------------------------------------------------

def main():
    began = time.perf_counter()
    plot.apply_style()
    cfg.METRICS_DIR.mkdir(parents=True, exist_ok=True)

    per_seed, statuses, episode_counts = {}, {}, []
    for seed in cfg.EXPERIMENT_SEEDS:
        training, status, split = validated_training_telemetry(seed)
        per_seed[seed], statuses[seed] = training, status
        counts = split.groupby(["split", "episode_type"]).size().rename("episodes")
        episode_counts.append(counts.reset_index().assign(seed=seed))
    episode_counts = pd.concat(episode_counts)
    pooled = pd.concat(per_seed.values(), ignore_index=True)
    main_seed = per_seed[cfg.RANDOM_SEED]

    # 1. Summary table (seed 42)
    summary = summary_by_state_cause(main_seed[main_seed.cause != cfg.NO_CAUSE])
    summary.to_csv(cfg.METRICS_DIR / "e1_summary_by_state_cause.csv", index=False)

    # 2. Separability: pooled and per-link baseline-relative, every seed
    rows = []
    for seed, training in per_seed.items():
        rows.append(separability(training, "pooled").assign(seed=seed))
        rows.append(separability(baseline_relative(training), "baseline_relative").assign(seed=seed))
    all_seeds = pd.concat(rows, ignore_index=True)
    all_seeds[all_seeds.seed == cfg.RANDOM_SEED].drop(columns="seed").to_csv(
        cfg.METRICS_DIR / "e1_separability_seed42.csv", index=False)
    across = (all_seeds.groupby(["view", "cause", "state", "metric"], sort=False)
              [["outside_reference_pct", "standardised_difference"]].agg(["min", "max"]))
    across.columns = [f"{a}_{b}" for a, b in across.columns]
    across.reset_index().to_csv(cfg.METRICS_DIR / "e1_separability_across_seeds.csv", index=False)

    # 3. Spearman correlation per cause (seed 42)
    correlations = {c: main_seed.loc[main_seed.cause == c, METRICS].corr(method="spearman")
                    for c in cfg.CAUSES}

    # 4. Rising phase, five seeds pooled; episodes keyed by (seed, episode_id)
    relative = pd.concat([baseline_relative(t) for t in per_seed.values()], ignore_index=True)
    window = rising_window(relative)
    summaries = rising_summaries(window)
    degrading = pooled[pooled.episode_type != cfg.STABLE].drop_duplicates(EPISODE_KEY)
    excluded = len(degrading) - len(summaries)
    rows = []
    for cause in cfg.CAUSES:
        for metric in METRICS:
            for measure in ("level", "speed"):
                column = f"{measure}_{metric}"
                group = summaries[summaries.cause == cause]
                worsening = group.loc[group.episode_type == cfg.WORSENING, column].dropna()
                recovering = group.loc[group.episode_type == cfg.RECOVERING, column].dropna()
                per_seed_p = [probability_of_superiority(
                    group.loc[(group.seed == s) & (group.episode_type == cfg.WORSENING), column].dropna(),
                    group.loc[(group.seed == s) & (group.episode_type == cfg.RECOVERING), column].dropna())
                    for s in cfg.EXPERIMENT_SEEDS]
                rows.append({"cause": cause, "metric": metric, "measure": measure,
                             "p_superiority_pooled": probability_of_superiority(worsening, recovering),
                             "p_superiority_seed_min": np.nanmin(per_seed_p),
                             "p_superiority_seed_max": np.nanmax(per_seed_p),
                             "median_worsening": worsening.median(),
                             "median_recovering": recovering.median(),
                             "episodes_worsening": len(worsening),
                             "episodes_recovering": len(recovering)})
    superiority = pd.DataFrame(rows)
    superiority.to_csv(cfg.METRICS_DIR / "e1_rising_phase_comparison.csv", index=False)

    figures = [figure_telemetry_by_state(main_seed[main_seed.cause != cfg.NO_CAUSE]),
               figure_correlation(correlations),
               figure_rising_phase(summaries, superiority)]

    # Console report
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 200)
    print("Validation status per seed:", statuses)
    print("\nEpisodes per seed by split and type:")
    print(episode_counts.pivot_table(index=["split", "episode_type"], columns="seed",
                                     values="episodes").to_string())
    print(f"\nTraining episodes analysed: {pooled.drop_duplicates(EPISODE_KEY).shape[0]} "
          f"(seed 42: {main_seed.episode_id.nunique()})")
    print(f"Rising-phase episodes: {len(summaries)}; excluded (never reached "
          f"{cfg.RISING_WINDOW_END_SEVERITY}): {excluded}")
    print("\nSpearman correlation (seed 42):")
    for cause, matrix in correlations.items():
        print(cause); print(matrix.round(2).to_string())
    print("\nFigures:", *[p.name for p in figures])
    print(f"Runtime: {time.perf_counter() - began:.1f} s")


if __name__ == "__main__":
    main()
