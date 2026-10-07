"""Simple descriptive statistics for Experiment 1 (network behaviour).

Episodes are identified by EPISODE_KEY = (seed, episode_id). Telemetry passed to these
functions must therefore carry a `seed` column, even when it holds a single seed.
"""

import numpy as np
import pandas as pd

from netsense import config as cfg

EPISODE_KEY = ["seed", "episode_id"]
METRICS = cfg.FEATURE_COLUMNS


def summary_by_state_cause(telemetry):
    """Mean, std, median and quartiles per (cause, state, metric); share of minutes with loss."""
    long = telemetry.melt(id_vars=["cause", "state"], value_vars=METRICS, var_name="metric")
    grouped = long.groupby(["cause", "metric", "state"])["value"]
    table = grouped.agg(mean="mean", std="std", median="median",
                        p25=lambda v: v.quantile(0.25), p75=lambda v: v.quantile(0.75),
                        minutes="size").reset_index()
    any_loss = (telemetry.assign(any_loss=telemetry.packet_loss_pct > 0)
                .groupby(["cause", "state"])["any_loss"].mean().mul(100).rename("pct_minutes_with_loss"))
    table = table.merge(any_loss.reset_index(), on=["cause", "state"], how="left")
    table.loc[table.metric != "packet_loss_pct", "pct_minutes_with_loss"] = np.nan
    return table


def outside_reference_share(normal_values, other_values, percentiles=cfg.NORMAL_REFERENCE_PERCENTILES):
    """% of `other_values` outside the empirical NORMAL reference range.

    The range is the given low/high percentiles of `normal_values`. If the two sets came
    from the same distribution, about (low + 100 - high)% would fall outside by chance.
    """
    low, high = np.percentile(normal_values, percentiles)
    other = np.asarray(other_values)
    return 100.0 * np.mean((other < low) | (other > high))


def standardised_difference(normal_values, other_values):
    """(mean of other - mean of normal) / standard deviation of normal."""
    normal = np.asarray(normal_values, dtype=float)
    spread = normal.std(ddof=1)
    if spread == 0:
        return np.nan
    return (np.mean(other_values) - normal.mean()) / spread


def baseline_relative(telemetry, window=cfg.BASELINE_WINDOW_MIN):
    """Each metric divided by the median of its own episode's first `window` minutes.

    Uses only that episode's own data (episodes keyed by (seed, episode_id)). Possible here
    because every simulated episode is healthy for at least its first 60 minutes.
    """
    baseline = (telemetry[telemetry.minute < window]
                .groupby(EPISODE_KEY)[METRICS].median())
    aligned = baseline.reindex(pd.MultiIndex.from_frame(telemetry[EPISODE_KEY])).to_numpy()
    values = telemetry[METRICS].to_numpy()
    # A ratio to a baseline of 0 is undefined (common for packet loss) -> NaN, reported.
    safe = np.where(aligned > 0, aligned, 1.0)
    ratios = np.where(aligned > 0, values / safe, np.nan)
    return telemetry.assign(**{m: ratios[:, i] for i, m in enumerate(METRICS)})


def separability(telemetry, view_name):
    """Outside-reference share and standardised difference, EARLY and DEGRADED vs NORMAL.

    The NORMAL reference is every NORMAL minute of every episode (stable ones included):
    the network-wide normal a single threshold would face. Undefined (NaN) values are
    left out and the minutes actually used are reported.
    """
    rows = []
    normal = telemetry[telemetry.state == cfg.NORMAL]
    for cause in cfg.CAUSES:
        for state in (cfg.EARLY_DEGRADATION, cfg.DEGRADED):
            other = telemetry[(telemetry.cause == cause) & (telemetry.state == state)]
            for metric in METRICS:
                n, o = normal[metric].dropna(), other[metric].dropna()
                rows.append({"view": view_name, "cause": cause, "state": state, "metric": metric,
                             "outside_reference_pct": outside_reference_share(n, o),
                             "standardised_difference": standardised_difference(n, o),
                             "normal_minutes": len(n), "state_minutes": len(o)})
    return pd.DataFrame(rows)


def rising_window(telemetry, end_severity=cfg.RISING_WINDOW_END_SEVERITY):
    """Rows from degradation onset up to and including the first minute severity reaches `end_severity`.

    Onset is the first minute with severity above 0. The same rule applies to worsening
    and recovering episodes; stable episodes have no rising window, and an episode that
    never reaches `end_severity` is excluded (its window would run into its recovery).
    Hidden severity defines the window for analysis only. Episodes keyed by (seed, episode_id).
    """
    degrading = telemetry[telemetry.episode_type != cfg.STABLE].sort_values(EPISODE_KEY + ["minute"])
    started = degrading.groupby(EPISODE_KEY)["hidden_severity"].transform(lambda s: (s > 0).cummax())
    reached = degrading.groupby(EPISODE_KEY)["hidden_severity"].transform(
        lambda s: (s >= end_severity).cummax().shift(fill_value=False))
    ever_reached = degrading.groupby(EPISODE_KEY)["hidden_severity"].transform(
        lambda s: (s >= end_severity).any())
    return degrading[started & ~reached & ever_reached]


def rising_summaries(rising_rows, level_window=cfg.RISING_LEVEL_WINDOW_MIN):
    """Per-episode level (median of the last minutes) and rise speed (change per minute)."""
    def summarise(episode):
        level = episode[METRICS].tail(level_window).median()
        start = episode[METRICS].head(level_window).median()
        minutes = max(len(episode) - level_window, 1)
        speed = (level - start) / minutes
        return pd.concat([level.add_prefix("level_"), speed.add_prefix("speed_"),
                          pd.Series({"window_minutes": len(episode)})])
    keys = EPISODE_KEY + ["episode_type", "cause"]
    return rising_rows.groupby(keys).apply(summarise, include_groups=False).reset_index()


def probability_of_superiority(higher_group, lower_group):
    """Chance that a random value from `higher_group` exceeds one from `lower_group`.

    0.5 means no difference; ties count as half. Computed over every pair of values.
    """
    a = np.asarray(higher_group, dtype=float)[:, None]
    b = np.asarray(lower_group, dtype=float)[None, :]
    return float(np.mean((a > b) + 0.5 * (a == b)))
