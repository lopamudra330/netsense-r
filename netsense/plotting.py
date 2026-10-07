"""Shared figure style for NetSense-R.

One consistent look for every figure: fixed state order, one colour per degradation
cause (colour-blind-safe pair, validated), recessive grid, text in neutral ink, and a
caption stating that the data is synthetic.
"""

import matplotlib

matplotlib.use("Agg")  # write files; no display needed
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from netsense import config as cfg

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

CAUSE_COLOURS = {cfg.CONGESTION: "#2a78d6", cfg.LINK_QUALITY: "#eb6834"}
CAUSE_LABELS = {cfg.CONGESTION: "Congestion", cfg.LINK_QUALITY: "Link-quality degradation"}
STATE_LABELS = {cfg.NORMAL: "Normal", cfg.EARLY_DEGRADATION: "Early",
                cfg.DEGRADED: "Degraded", cfg.SEVERE_DEGRADATION: "Severe"}
METRIC_LABELS = {
    "latency_ms": "Latency (ms)",
    "jitter_ms": "Jitter (ms)",
    "packet_loss_pct": "Packet loss (%)",
    "error_rate_pct": "Frame error rate (%)",
    "retransmission_rate_pct": "Retransmission rate (%)",
    "throughput_mbps": "Throughput (Mbit/s)",
}
SHORT_METRIC_LABELS = {
    "latency_ms": "Latency", "jitter_ms": "Jitter", "packet_loss_pct": "Loss",
    "error_rate_pct": "Error rate", "retransmission_rate_pct": "Retransmissions",
    "throughput_mbps": "Throughput",
}

# Diverging scale for correlations: blue (-1) <-> neutral grey (0) <-> red (+1).
DIVERGING = LinearSegmentedColormap.from_list(
    "netsense_diverging", ["#184f95", "#86b6ef", "#f0efec", "#ef9a99", "#b3302f"])

SYNTHETIC_NOTE = "Synthetic telemetry; not measurements from any real network."


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK_SECONDARY,
        "xtick.color": INK_SECONDARY, "ytick.color": INK_SECONDARY,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def save(fig, filename, caption):
    """Add the caption (with the synthetic-data note) and write a PNG to results/figures."""
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Placed below the axes (va="top" at y=0); bbox_inches="tight" makes room for it.
    fig.text(0.01, -0.01, f"{caption} {SYNTHETIC_NOTE}", ha="left", va="top",
             fontsize=7.5, color=INK_SECONDARY, wrap=True)
    path = cfg.FIGURES_DIR / filename
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path
