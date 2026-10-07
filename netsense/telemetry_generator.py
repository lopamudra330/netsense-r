"""Synthetic telemetry generator for NetSense-R.

THIS DATA IS SYNTHETIC. It is designed for methodological exploration and does not
represent measurements from any specific production network.

How one episode is built
------------------------
1. A hidden *severity* path (0 = healthy, 1 = fully degraded) is drawn for the episode.
2. Severity drives the link's underlying conditions each minute:
   - congestion:   severity raises traffic demand (utilisation);
   - link quality: severity raises the frame error rate.
   Harmless traffic and interference bursts are added on top; they never change
   severity or labels.
3. Those conditions are "measured" the way a monitoring system would:
   - latency, jitter and packet loss come from 600 simulated probe packets per minute;
   - error rate, retransmission rate and throughput come from simulated link counters.
4. The state label is derived from the hidden severity only, never from telemetry.

All parameter values live in netsense/config.py (frozen block).
"""

import time

import numpy as np
import pandas as pd

from netsense import config as cfg

TELEMETRY_COLUMNS = [
    "timestamp",
    "episode_id",
    "minute",
    "episode_type",
    "cause",
    *cfg.FEATURE_COLUMNS,
    "connection_state",
    "hidden_severity",
    "state",
]


# --- Episode plan -----------------------------------------------------------

def build_episode_plan(rng):
    """Return the list of (episode_type, cause) pairs, in random order."""
    plan = [(cfg.STABLE, cfg.NO_CAUSE)] * cfg.N_STABLE_EPISODES
    for cause in cfg.CAUSES:
        plan += [(cfg.WORSENING, cause)] * cfg.N_WORSENING_EPISODES_PER_CAUSE
        plan += [(cfg.RECOVERING, cause)] * cfg.N_RECOVERING_EPISODES_PER_CAUSE
    order = rng.permutation(len(plan))
    return [plan[i] for i in order]


# --- Hidden severity --------------------------------------------------------

def severity_to_state(severity):
    """Map hidden severity values to state labels using the fixed band edges."""
    severity = np.asarray(severity)
    band = np.digitize(severity, cfg.STATE_BAND_EDGES)  # 0, 1, 2 or 3
    return np.array(cfg.STATES)[band]


def severity_path(episode_type, rng, variability=1.0):
    """Draw the hidden severity for every minute of one episode.

    Returns (severity array, info dict). The info dict records the drawn timing
    parameters so they can be checked and reported.
    """
    n = cfg.EPISODE_MINUTES
    minutes = np.arange(n)
    if episode_type == cfg.STABLE:
        return np.zeros(n), {"onset_minute": np.nan, "time_to_full_min": np.nan,
                             "recovery_peak": np.nan}

    onset = int(rng.integers(cfg.ONSET_RANGE_MIN[0], cfg.ONSET_RANGE_MIN[1] + 1))
    time_to_full = rng.uniform(*cfg.TIME_TO_FULL_SEVERITY_RANGE_MIN)
    since_onset = minutes - onset
    rising = since_onset / time_to_full  # steady rise: 0 at onset, 1 after D minutes

    if episode_type == cfg.WORSENING:
        ramp = np.clip(rising, 0.0, 1.0)
        active = since_onset >= 0
        peak = np.nan
    else:  # RECOVERING: rise at the same rate, hold at the peak, fall back to 0
        peak = rng.uniform(*cfg.RECOVERY_PEAK_RANGE)
        hold = rng.uniform(*cfg.RECOVERY_HOLD_RANGE_MIN)
        fall = rng.uniform(*cfg.RECOVERY_FALL_RANGE_MIN)
        peak_reached = onset + peak * time_to_full
        fall_starts = peak_reached + hold
        falling = peak * (1.0 - (minutes - fall_starts) / fall)
        ramp = np.where(minutes < peak_reached, rising,
                        np.where(minutes < fall_starts, peak, falling))
        ramp = np.clip(ramp, 0.0, 1.0)
        active = (since_onset >= 0) & (minutes < fall_starts + fall)

    # Wobble: each minute keeps 90% of the previous wiggle and adds a small new step.
    steps = rng.normal(0.0, cfg.WOBBLE_STEP_STD * variability, n)
    wobble = np.zeros(n)
    for t in range(1, n):
        wobble[t] = cfg.WOBBLE_PERSISTENCE * wobble[t - 1] + steps[t]

    severity = np.where(active, np.clip(ramp + wobble, 0.0, 1.0), 0.0)
    info = {"onset_minute": onset, "time_to_full_min": time_to_full,
            "recovery_peak": peak}
    return severity, info


# --- Harmless transients ----------------------------------------------------

def burst_mask(rng, start_chance, duration_range, size_range, n=cfg.EPISODE_MINUTES):
    """Place random bursts in an episode.

    Each minute has a small chance that a burst starts. A burst lasts a random
    number of minutes and has a random size. Where bursts overlap, the larger
    size applies.

    Returns (per-minute size array with 0 where there is no burst, list of bursts).
    """
    sizes = np.zeros(n)
    bursts = []
    for start in np.flatnonzero(rng.random(n) < start_chance):
        duration = int(rng.integers(duration_range[0], duration_range[1] + 1))
        size = rng.uniform(*size_range)
        end = min(start + duration, n)
        sizes[start:end] = np.maximum(sizes[start:end], size)
        bursts.append({"start": int(start), "duration": duration, "size": size})
    return sizes, bursts


# --- Underlying link conditions ---------------------------------------------

def link_conditions(severity, cause, link, rng, variability=1.0):
    """Turn hidden severity into the link's underlying per-minute conditions.

    `link` holds the episode's own baseline values (normal utilisation and normal
    frame error rate). Returns a dict of per-minute arrays.
    """
    n = len(severity)

    # Traffic demand as a share of capacity ("utilisation").
    traffic_bursts, _ = burst_mask(rng, cfg.TRAFFIC_BURST_START_CHANCE,
                                   cfg.TRAFFIC_BURST_DURATION_MIN,
                                   cfg.TRAFFIC_BURST_EXTRA_UTILISATION, n)
    utilisation = (link["normal_utilisation"]
                   + rng.normal(0.0, cfg.TRAFFIC_VARIATION_STD * variability, n)
                   + traffic_bursts)
    if cause == cfg.CONGESTION:
        utilisation += severity * (cfg.CONGESTION_PEAK_UTILISATION
                                   - link["normal_utilisation"])
    utilisation = np.maximum(utilisation, 0.0)

    # Frame error rate (a fraction, 0-1).
    interference, _ = burst_mask(rng, cfg.INTERFERENCE_BURST_START_CHANCE,
                                 cfg.INTERFERENCE_BURST_DURATION_MIN,
                                 cfg.INTERFERENCE_BURST_ERROR_FACTOR, n)
    frame_error = link["normal_frame_error"] * np.where(interference > 0, interference, 1.0)
    if cause == cfg.LINK_QUALITY:
        # Multiplies by about 2.3 for every 0.1 of severity (slow, then steep).
        frame_error = frame_error * cfg.FRAME_ERROR_GROWTH_PER_TENTH ** (10 * severity)
    frame_error = np.minimum(frame_error, 1.0)

    # Congestion drops: if 125 units arrive and only 100 fit, 25/125 = 20% is dropped.
    congestion_loss = np.where(utilisation > 1.0, 1.0 - 1.0 / np.maximum(utilisation, 1.0), 0.0)
    # Corrupted frames: some are re-sent by the link (late), the rest are lost.
    link_loss = (1.0 - cfg.LINK_RETRY_SHARE) * frame_error
    retry_chance = cfg.LINK_RETRY_SHARE * frame_error
    # A packet arrives only if it survives all three independent loss causes.
    total_loss = 1.0 - (1.0 - cfg.BACKGROUND_LOSS) * (1.0 - congestion_loss) * (1.0 - link_loss)

    # Average queueing delay grows slowly, then explosively, as the link fills up.
    capped_util = np.minimum(utilisation, cfg.QUEUE_UTILISATION_CAP)
    mean_queue_delay = cfg.QUEUE_DELAY_SCALE_MS * capped_util / (1.0 - capped_util)

    return {
        "utilisation": utilisation,
        "frame_error": frame_error,
        "congestion_loss": congestion_loss,
        "link_loss": link_loss,
        "retry_chance": retry_chance,
        "total_loss": total_loss,
        "mean_queue_delay_ms": mean_queue_delay,
    }


# --- Measurement --------------------------------------------------------------

def measure_minute_probes(base_delay_ms, conditions, rng):
    """Simulate 600 probe packets per minute and summarise them.

    Returns per-minute arrays: latency_ms (average delay of probes that arrived),
    jitter_ms (standard deviation of those delays) and packet_loss_pct.
    """
    n = len(conditions["total_loss"])
    shape = (n, cfg.PROBES_PER_MINUTE)

    # Queueing waits: many short, a few long (exponential); never beyond the buffer.
    queue = rng.exponential(1.0, shape) * conditions["mean_queue_delay_ms"][:, None]
    queue = np.minimum(queue, cfg.BUFFER_MAX_DELAY_MS)
    processing = rng.normal(0.0, cfg.PROBE_PROCESSING_NOISE_MS, shape)
    retried = rng.random(shape) < conditions["retry_chance"][:, None]
    delay = base_delay_ms + queue + processing + retried * cfg.LINK_RETRY_DELAY_MS

    lost = rng.random(shape) < conditions["total_loss"][:, None]
    arrived = np.where(lost, np.nan, delay)
    with np.errstate(invalid="ignore"):
        latency = np.nanmean(arrived, axis=1)
        jitter = np.nanstd(arrived, axis=1)
    loss_pct = lost.mean(axis=1) * 100.0
    return latency, jitter, loss_pct


def measure_counters(conditions, rng, variability=1.0):
    """Simulate link-counter readings: error rate, retransmissions, throughput.

    Counters see huge numbers of frames, so sampling noise is negligible; instead the
    underlying rates fluctuate. That is modelled as multiplicative noise: a random
    factor near 1, so a value varies by a percentage of itself.
    """
    n = len(conditions["total_loss"])
    error_noise = np.exp(rng.normal(0.0, cfg.ERROR_COUNTER_NOISE * variability, n))
    retx_noise = np.exp(rng.normal(0.0, cfg.RETRANSMISSION_COUNTER_NOISE * variability, n))

    error_rate_pct = np.minimum(conditions["frame_error"] * error_noise * 100.0, 100.0)
    retransmission_pct = np.minimum(conditions["total_loss"] * retx_noise * 100.0, 100.0)

    # Delivered traffic: never more than capacity, minus traffic lost to errors and
    # background loss (congestion drops are the excess above capacity).
    carried = np.minimum(conditions["utilisation"], 1.0) * cfg.LINK_CAPACITY_MBPS
    throughput = carried * (1.0 - conditions["link_loss"]) * (1.0 - cfg.BACKGROUND_LOSS)
    return error_rate_pct, retransmission_pct, throughput


# --- Episodes and dataset ---------------------------------------------------

def generate_episode(episode_id, episode_type, cause, rng, variability=1.0):
    """Generate one 6-hour episode. Returns (telemetry DataFrame, episode info dict)."""
    link = {
        "base_delay_ms": rng.uniform(*cfg.BASE_DELAY_RANGE_MS),
        "normal_utilisation": rng.uniform(*cfg.NORMAL_UTILISATION_RANGE),
        "normal_frame_error": rng.uniform(*cfg.NORMAL_FRAME_ERROR_RANGE_PCT) / 100.0,
    }
    severity, timing = severity_path(episode_type, rng, variability)
    conditions = link_conditions(severity, cause, link, rng, variability)
    latency, jitter, loss_pct = measure_minute_probes(link["base_delay_ms"], conditions, rng)
    error_pct, retx_pct, throughput = measure_counters(conditions, rng, variability)

    n = cfg.EPISODE_MINUTES
    start = (pd.Timestamp(cfg.SYNTHETIC_START_TIME)
             + pd.Timedelta(seconds=episode_id * n * cfg.SAMPLE_INTERVAL_SECONDS))
    telemetry = pd.DataFrame({
        "timestamp": pd.date_range(start, periods=n,
                                   freq=pd.Timedelta(seconds=cfg.SAMPLE_INTERVAL_SECONDS)),
        "episode_id": episode_id,
        "minute": np.arange(n),
        "episode_type": episode_type,
        "cause": cause,
        "latency_ms": latency,
        "jitter_ms": jitter,
        "packet_loss_pct": loss_pct,
        "error_rate_pct": error_pct,
        "retransmission_rate_pct": retx_pct,
        "throughput_mbps": throughput,
        "connection_state": np.where(loss_pct >= cfg.UNSTABLE_LOSS_PCT, "UNSTABLE", "UP"),
        "hidden_severity": severity,
        "state": severity_to_state(severity),
    })
    info = {"episode_id": episode_id, "episode_type": episode_type, "cause": cause,
            **link, **timing}
    return telemetry, info


def generate_dataset(seed=cfg.RANDOM_SEED, variability=1.0):
    """Generate the full synthetic dataset for one seed.

    Returns (telemetry DataFrame, episodes DataFrame with one row per episode).
    """
    rng = np.random.default_rng(seed)
    frames, infos = [], []
    for episode_id, (episode_type, cause) in enumerate(build_episode_plan(rng)):
        telemetry, info = generate_episode(episode_id, episode_type, cause, rng, variability)
        frames.append(telemetry)
        infos.append(info)
    return pd.concat(frames, ignore_index=True)[TELEMETRY_COLUMNS], pd.DataFrame(infos)


def save_dataset(telemetry, episodes, seed, directory=cfg.GENERATED_DATA_DIR):
    """Write telemetry and episode tables as CSV files. Returns the telemetry path."""
    directory.mkdir(parents=True, exist_ok=True)
    telemetry_path = directory / f"telemetry_seed{seed}.csv"
    telemetry.to_csv(telemetry_path, index=False)
    episodes.to_csv(directory / f"episodes_seed{seed}.csv", index=False)
    return telemetry_path


if __name__ == "__main__":
    began = time.perf_counter()
    telemetry, episodes = generate_dataset(cfg.RANDOM_SEED)
    path = save_dataset(telemetry, episodes, cfg.RANDOM_SEED)
    print(f"Generated {len(telemetry):,} rows from {len(episodes)} episodes "
          f"in {time.perf_counter() - began:.1f} s -> {path}")
