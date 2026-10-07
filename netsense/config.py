"""Central configuration for NetSense-R.

Every constant used by the project is defined here, with a comment explaining it,
so that no unexplained "magic numbers" appear elsewhere in the code.

Step 1 contains only the names fixed by the research design. Telemetry-generator
parameters are added in Step 2, where each value is justified.
"""

from pathlib import Path

# --- Reproducibility --------------------------------------------------------
# A fixed seed makes every "random" result repeatable: same seed, same data.
RANDOM_SEED = 42

# --- Network states (ordered from healthy to failed) ------------------------
NORMAL = "NORMAL"
EARLY_DEGRADATION = "EARLY_DEGRADATION"
DEGRADED = "DEGRADED"
SEVERE_FAILURE = "SEVERE_FAILURE"
STATES = [NORMAL, EARLY_DEGRADATION, DEGRADED, SEVERE_FAILURE]

# --- Degradation causes -----------------------------------------------------
# Congestion: traffic approaches link capacity, so queues build up.
CONGESTION = "congestion"
# Link quality: the transmission link itself deteriorates, so frames are corrupted.
LINK_QUALITY = "link_quality"
CAUSES = [CONGESTION, LINK_QUALITY]

# --- Sampling ---------------------------------------------------------------
# One aggregated telemetry sample per minute, similar to periodic counter polling.
SAMPLE_INTERVAL_SECONDS = 60

# --- Paths ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DATA_DIR = PROJECT_ROOT / "data" / "generated"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
