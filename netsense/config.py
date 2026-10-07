"""Central configuration for NetSense-R.

Every constant used by the project is defined here, with a comment explaining it,
so that no unexplained "magic numbers" appear elsewhere in the code.
"""

from pathlib import Path

# --- Reproducibility --------------------------------------------------------
# A fixed seed makes every "random" result repeatable: same seed, same data.
RANDOM_SEED = 42
# The full study is repeated with these seeds to show how much results vary by chance.
EXPERIMENT_SEEDS = [42, 43, 44, 45, 46]

# --- Network states (ordered from healthy to most degraded) -----------------
NORMAL = "NORMAL"
EARLY_DEGRADATION = "EARLY_DEGRADATION"
DEGRADED = "DEGRADED"
SEVERE_DEGRADATION = "SEVERE_DEGRADATION"
STATES = [NORMAL, EARLY_DEGRADATION, DEGRADED, SEVERE_DEGRADATION]

# Entry into SEVERE_DEGRADATION is the "critical-degradation threshold": a modelling
# convention used as the endpoint for lead-time measurement. It is not literal link
# failure; the simulated link may still carry traffic.
CRITICAL_STATE = SEVERE_DEGRADATION

# --- Degradation causes -----------------------------------------------------
# Congestion: traffic approaches or exceeds link capacity, so queues build up.
CONGESTION = "congestion"
# Link quality: the transmission link itself deteriorates, so frames are corrupted.
LINK_QUALITY = "link_quality"
CAUSES = [CONGESTION, LINK_QUALITY]
NO_CAUSE = "none"  # used for stable episodes

# --- Episode types ----------------------------------------------------------
STABLE = "stable"          # normal operation with harmless bursts only
WORSENING = "worsening"    # degrades until it reaches severe degradation
RECOVERING = "recovering"  # degrades, then returns to normal without reaching severe
EPISODE_TYPES = [STABLE, WORSENING, RECOVERING]

# --- Sampling ---------------------------------------------------------------
# One aggregated telemetry sample per minute, similar to periodic counter polling.
SAMPLE_INTERVAL_SECONDS = 60

# --- Detector inputs (allow-list) -------------------------------------------
# Detectors may use ONLY these observed telemetry columns (and summaries of them).
# Ground truth (state, hidden_severity), bookkeeping (episode_id, episode_type,
# cause, minute, timestamp) and the reactive status (connection_state) are excluded.
FEATURE_COLUMNS = [
    "latency_ms",
    "jitter_ms",
    "packet_loss_pct",
    "error_rate_pct",
    "retransmission_rate_pct",
    "throughput_mbps",
]

# =============================================================================
# FROZEN GENERATOR PARAMETERS — frozen 2026-10-08, before any detector existed.
#
# These values must not be changed to improve (or worsen) detector results.
# A change is allowed only to fix a modelling or code error, must be explained
# before it is made, and must be recorded in the change log of
# docs/methodology.md. Rationale for each value: docs/methodology.md and
# data/README.md.
# =============================================================================

# --- Study size ---
N_STABLE_EPISODES = 80
N_WORSENING_EPISODES_PER_CAUSE = 60
N_RECOVERING_EPISODES_PER_CAUSE = 30
TEST_FRACTION = 0.5                       # whole episodes go to train or test
VARIABILITY_MULTIPLIERS = [0.5, 1.0, 2.0]  # sensitivity analysis (1.0 = default)

# --- Time ---
EPISODE_MINUTES = 360                     # 6-hour episodes, 1 sample per minute
ONSET_RANGE_MIN = (60, 120)               # degradation onset, minutes into episode
TIME_TO_FULL_SEVERITY_RANGE_MIN = (60, 180)  # "D": onset -> severity 1
SYNTHETIC_START_TIME = "2000-01-01 00:00"  # arbitrary clock; no calendar meaning

# --- Hidden severity ---
WOBBLE_PERSISTENCE = 0.9    # share of last minute's wobble that carries over
WOBBLE_STEP_STD = 0.01      # size of each minute's new random wobble step
STATE_BAND_EDGES = (0.10, 0.40, 0.70)  # NORMAL | EARLY | DEGRADED | SEVERE
RECOVERY_PEAK_RANGE = (0.25, 0.60)     # recovering episodes stay below 0.70
RECOVERY_HOLD_RANGE_MIN = (10, 30)
RECOVERY_FALL_RANGE_MIN = (30, 90)

# --- Link ---
LINK_CAPACITY_MBPS = 100.0
BASE_DELAY_RANGE_MS = (10.0, 30.0)       # propagation + processing, per episode
NORMAL_UTILISATION_RANGE = (0.30, 0.60)  # normal share of capacity in use, per episode
TRAFFIC_VARIATION_STD = 0.05             # minute-to-minute utilisation variation

# --- Congestion mechanism ---
CONGESTION_PEAK_UTILISATION = 1.25  # offered load at severity 1 (125% of capacity)
QUEUE_DELAY_SCALE_MS = 3.0          # queueing delay = scale * u / (1 - u)
QUEUE_UTILISATION_CAP = 0.97        # beyond this the buffer is treated as effectively full
BUFFER_MAX_DELAY_MS = 100.0         # no packet waits longer than the buffer holds

# --- Link-quality mechanism ---
NORMAL_FRAME_ERROR_RANGE_PCT = (0.005, 0.02)  # per episode
FRAME_ERROR_GROWTH_PER_TENTH = 2.3  # error rate multiplies by this per 0.1 of severity
LINK_RETRY_SHARE = 0.5              # share of corrupted frames re-sent by the link
LINK_RETRY_DELAY_MS = 20.0          # extra delay of a re-sent frame

# --- Measurement ---
PROBES_PER_MINUTE = 600             # 10 probe packets per second
PROBE_PROCESSING_NOISE_MS = 0.5
BACKGROUND_LOSS = 0.001             # 0.1% loss present at all times
ERROR_COUNTER_NOISE = 0.3           # typical log-spread of the error-rate counter
RETRANSMISSION_COUNTER_NOISE = 0.2  # typical log-spread of the retransmission counter

# --- Harmless transients (never change severity or labels) ---
TRAFFIC_BURST_START_CHANCE = 1 / 90       # per minute
TRAFFIC_BURST_DURATION_MIN = (2, 10)
TRAFFIC_BURST_EXTRA_UTILISATION = (0.20, 0.45)
INTERFERENCE_BURST_START_CHANCE = 1 / 180  # per minute
INTERFERENCE_BURST_DURATION_MIN = (1, 5)
INTERFERENCE_BURST_ERROR_FACTOR = (10.0, 50.0)

# --- Reactive status ---
UNSTABLE_LOSS_PCT = 5.0  # connection_state = UNSTABLE when probe loss >= this

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DATA_DIR = PROJECT_ROOT / "data" / "generated"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
