# Data

**This dataset is synthetic and designed for methodological exploration; it does not
represent measurements from a specific production network.**

It models a generic IP access/backhaul link. It does not represent, and is not
calibrated to, any smart-metering, DCC, Arqiva or other proprietary system. The numeric
values are illustrative orders of magnitude.

## Generating the data

```bash
python -m netsense.telemetry_generator
```

This writes two files to `data/generated/`. They are not committed to git, because they
can be recreated exactly from the fixed seed:

- `telemetry_seed42.csv`: one row per link per minute (93,600 rows)
- `episodes_seed42.csv`: one row per episode, with its drawn baseline and timing values

## Telemetry columns

### Observed telemetry: the only columns detectors may use

| Column | Unit | Source | Meaning |
|---|---|---|---|
| `latency_ms` | ms | 600 probes per minute | Average delay of the probes that arrived |
| `jitter_ms` | ms | 600 probes per minute | Standard deviation of those delays. This is not the RFC 3550 interarrival jitter |
| `packet_loss_pct` | % | 600 probes per minute | Share of probes lost, measured in steps of ≈ 0.17% |
| `error_rate_pct` | % | Link counter | Share of frames that failed their error check |
| `retransmission_rate_pct` | % | Link counter | Share of segments re-sent by senders |
| `throughput_mbps` | Mbit/s | Link counter | Traffic delivered across the 100 Mbit/s link |

### Not detector inputs

| Column | Meaning |
|---|---|
| `timestamp` | Synthetic clock starting at 2000-01-01 with no calendar meaning; episodes are laid end to end |
| `episode_id` | Which episode (independent simulated link) the row belongs to |
| `minute` | Minute within the episode (0–359) |
| `episode_type` | `stable`, `worsening` or `recovering` |
| `cause` | `congestion`, `link_quality` or `none` |
| `connection_state` | `UNSTABLE` if probe loss ≥ 5% in that minute, otherwise `UP`. A reactive status derived from observed loss |
| `hidden_severity` | Ground truth: underlying condition from 0 to 1 |
| `state` | Ground-truth label, derived only from `hidden_severity` |

## How each variable behaves

| Variable | Congestion | Link-quality degradation | Why |
|---|---|---|---|
| Latency | Rises strongly, then plateaus when the buffer is effectively full | Rises only a few ms | Packets queue in a busy link; only a minority of corrupted frames are re-sent late |
| Jitter | Rises early with queueing | Rises | Queues fluctuate; partial retries make delay uneven |
| Packet loss | Near zero until demand exceeds capacity, then rises sharply | Rises steadily | Full buffers drop the excess; corrupted frames are discarded |
| Error rate | Unchanged | Rises steeply (in relative terms, from a tiny baseline) | Congestion does not corrupt bits; a degrading link does |
| Retransmissions | Follow loss | Follow loss | Senders re-send what was lost |
| Throughput | Rises to capacity, then plateaus | Falls | A full link carries maximum traffic; lost frames are not delivered |

Harmless traffic and interference bursts occur in every episode and never change the
labels. The full rationale, the label convention and the study design are in
[`docs/methodology.md`](../docs/methodology.md). Parameter values are in the frozen block
of [`netsense/config.py`](../netsense/config.py).
