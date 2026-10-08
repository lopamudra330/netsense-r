# Methodology

This document records the research questions and experimental plan for NetSense-R.
The plan was written **before** any detector was built or any model trained. Changes
to the plan are recorded in the [change log](#change-log), with the reason for each.

## 1. Motivation

Operational monitoring of communication systems is often reactive: an alarm fires once
a service is already badly affected, and engineers then work backwards to find out why.
NetSense-R asks whether the telemetry a link already produces (delay, loss,
retransmissions, throughput) shows a recognisable *signature* of developing degradation
early enough to act on.

The engineering motivation comes from my professional experience investigating
communication failures in the UK smart-metering domain. **The study itself does not use,
model or represent any smart-metering, DCC, Arqiva or other proprietary or production
system.** It models a generic IP access/backhaul link.

## 2. Research questions

### Primary question

> In a controlled, synthetic model of gradual communication-link degradation, how
> reliably can (a) conventional threshold monitoring and (b) simple interpretable
> machine-learning models detect degradation, how much warning can they give before the
> link enters a severe degradation state, and at what cost in false alarms?

The question is methodological. Because the data is synthetic, the answers describe how
the detectors behave **under the stated assumptions of the simulation**. They are not
claims about how detectable degradation is in real networks.

### Supporting questions

| | Question | Addressed by |
|---|---|---|
| **RQ1** | Which telemetry variables change earliest and most clearly as a link moves from normal operation into early degradation, and does this depend on the *cause* of degradation? | Experiments 1 and 4 |
| **RQ2** | When is simple threshold logic sufficient, and when (if ever) do Logistic Regression or Random Forest add detection value, especially for early degradation? | Experiments 2 and 3 |
| **RQ3** | What kinds of errors remain (false alarms, missed early degradation, confusion between neighbouring states), how well do detectors generalise to a degradation cause they were not trained on, and why are these cases difficult? | Experiments 3 and 4, robustness check, failure-case analysis |

### Future research question (not implemented)

> **Prognosis.** Once early degradation is observed, can we determine whether the link
> will recover or continue towards severe degradation?

The recovering episodes in this study make the question visible, but answering it is
outside the scope of this project.

## 3. Terminology

- **Degradation detection**, not anomaly detection. The ML experiments are
  *supervised*: models learn from examples labelled with the network state.
  Unsupervised anomaly detection, which learns without labels and could in principle
  flag unfamiliar degradation types, is identified as future work.
- **Network states**: `NORMAL`, `EARLY_DEGRADATION`, `DEGRADED`, `SEVERE_DEGRADATION`.
  The most degraded state is called *severe degradation* rather than *failure*
  because the simulated link may still be carrying traffic.
- **Critical-degradation threshold**: entry into `SEVERE_DEGRADATION`, which is the
  first minute in which hidden severity reaches 0.70. *This is a modelling convention
  used as the endpoint for lead-time measurement. It does not represent literal link
  failure.*
- **Degradation causes**:
  - *Congestion*: offered traffic approaches or exceeds link capacity, so queues
    build up in network buffers.
  - *Link-quality degradation*: the transmission link itself deteriorates (for
    example interference or a failing interface), so frames are corrupted.

## 4. Synthetic telemetry model

*All telemetry is synthetic and designed for methodological exploration; it does not
represent measurements from a specific production network.* The parameter values are
in the frozen block of [`netsense/config.py`](../netsense/config.py). The data
dictionary is in [`data/README.md`](../data/README.md).

### 4.1 Episodes and hidden severity

The data consists of independent **episodes**. Each episode is a 6-hour observation of
one simulated link, sampled once per minute. Each episode has a hidden **severity**
from 0 (healthy) to 1 (fully degraded) that changes over time. No detector ever sees it.

| Episode type | Severity path |
|---|---|
| Stable | Always 0. Harmless bursts still occur |
| Worsening | 0 until a random onset (60–120 min), then rises steadily to 1 over a random 60–180 min, then stays at 1 |
| Recovering | Rises at the same kind of rate to a random peak (0.25–0.60), holds for 10–30 min, then falls back to 0 over 30–90 min |

A small *wobble* is added after onset. Each minute it keeps 90% of the previous minute's
wiggle and adds a small new random step, so the path is not perfectly smooth.

The rising phase of a recovering episode follows the same rule as a worsening one.
**At the time of an early alert, the two can look the same.** This is deliberate: it is
why prognosis is a separate, harder problem.

### 4.2 Labels come only from hidden severity

| Severity | State |
|---|---|
| below 0.10 | NORMAL |
| 0.10 to 0.40 | EARLY_DEGRADATION |
| 0.40 to 0.70 | DEGRADED |
| 0.70 and above | SEVERE_DEGRADATION |

The cut-offs are conventions, not physics. A small NORMAL band absorbs negligible
severity, and the rest of the range is split into three equal bands. Labels describe
the link's *persistent underlying condition*, not momentary user experience. That is why
a 5-minute traffic burst is labelled NORMAL.

### 4.3 Two degradation mechanisms

**Congestion:** severity raises traffic demand, from the link's normal utilisation up to
125% of capacity at severity 1.
- Average queueing delay follows `3 ms × u / (1 − u)`, where *u* is utilisation. It grows
  slowly and then explosively as the link fills: 3 ms at 50% busy, 12 ms at 80%, 27 ms
  at 90%. Only the *shape* of this simple textbook queue model is used; the 3 ms scale
  is synthetic. Above 97% utilisation the buffer is treated as effectively full
  (100 ms maximum wait).
- Loss appears only when demand exceeds capacity: the excess is dropped. For example, if
  125 units arrive and only 100 fit, 25/125 = 20% are lost.
- **Throughput rises to link capacity and then plateaus.** The link is full, and the
  excess is dropped.
- The frame error rate is unchanged, because congestion does not corrupt bits.

**Link-quality degradation:** severity raises the frame error rate, which multiplies by
about 2.3 for every 0.1 of severity. This is slow and then steep, like the "cliff effect"
of digital links.
- Half of corrupted frames are re-sent by the link (that packet arrives 20 ms late), and
  half are lost.
- **Average latency changes only slightly**, by a few milliseconds even at severe
  degradation, because only a minority of packets are delayed. The signal is in error
  rate, loss, retransmissions and jitter.
- Throughput falls because lost traffic is not delivered.
- Traffic demand is unchanged.

**Common severity scale:** both mechanisms are set so that at severity 1 the link loses
about 20% of packets. This gives `SEVERE_DEGRADATION` a comparable service meaning for
both causes.

### 4.4 Measurement

- **Latency, jitter and packet loss** come from 600 simulated probe packets per minute.
  They are the average delay, the standard deviation of delay, and the share of probes
  lost. Loss can therefore only be measured in steps of 1/600 ≈ 0.17%.
- **Error rate, retransmission rate and throughput** come from simulated link counters.
  The underlying rates fluctuate from minute to minute; this is modelled with
  multiplicative noise (a random factor near 1).
- `connection_state` is UNSTABLE when probe loss in a minute is at least 5%. It is derived
  from *observed* loss, represents the reactive status alarm operations teams already
  see, and is never a detector input.

### 4.5 Harmless transients

Short **traffic bursts** (on average one per 90 min, lasting 2–10 min, adding 20–45% of
capacity) and **interference bursts** (on average one per 180 min, lasting 1–5 min,
multiplying the error rate by 10–50) occur in every episode. They never change severity
or labels. Both kinds exist so that each cause has realistic short-lived lookalikes.

## 5. Study size: why episodes, not rows

Consecutive minutes within an episode are strongly related, so **independent episodes,
not telemetry rows, are the unit of evidence**.

Each seed generates 260 episodes: 80 stable, and for each cause 60 worsening and 30
recovering. That is 93,600 rows; the row count follows from the design and is not a
target. Whole episodes are split 50/50 into training and testing, balanced across type
and cause. The test set therefore contains 30 worsening episodes per cause, 15 recovering
episodes per cause, and 40 stable episodes (10 days of normal operation).

**Why 30 test episodes per cause?** If a proportion such as "share of worsening episodes
warned in advance" is estimated from *n* independent episodes, its typical uncertainty
(standard error) is at most about 50% ÷ √n. With n = 30 that is about ±9 percentage
points, or roughly ±18 points for a 95% range. That is enough to separate large
differences (for example 40% vs 90%) but not small ones, which is appropriate for a
methodological study. Halving the uncertainty would need four times as many episodes.

**Why not far more synthetic episodes?** More episodes shrink *random* uncertainty but
cannot remove the main uncertainty, which is whether the simulation's assumptions are
right. A modest count also matches the scale a real study could collect, since real
degradation events are rare. The whole study is repeated over 5 seeds to show how much
results vary by chance.

The episode mix deliberately over-represents degradation compared with a real network.
False alarms per 24 hours are measured on normal time and are unaffected, but
**precision depends on how common degradation is**, so precision values do not
transfer to operational settings.

## 6. Study design safeguards

1. **Labels from hidden severity only**, never from observed telemetry.
2. **Feature allow-list.** Detectors may use only the six observed telemetry columns
   (and summaries of them). Timestamps, minute index, cause, episode identifiers, labels,
   hidden severity and `connection_state` are excluded. This allow-list, not the random
   onset, is the protection against time-based leakage. Onset randomisation only
   reduces an obvious temporal shortcut.
3. **Episode-level train/test split.** Neighbouring minutes never appear in both sets.
4. **Rolling features computed within each episode**, never across episode boundaries.
5. **Thresholds and any scaling fitted on training data only.**
6. **Harmless bursts and recovering episodes** so that early signs are not trivially
   informative.
7. **Five random seeds** and a **sensitivity analysis** at variability ×0.5, ×1 and ×2.
   The multiplier scales traffic variation, counter noise and severity wobble together.
8. **Generator parameters frozen** (2026-10-08) before any detector code existed.
9. **Telemetry validated before any detector sees it** (see section 10).

## 7. Experiments

| Exp | Question | Method | Metrics |
|---|---|---|---|
| 1. Behaviour | What does degradation look like in telemetry, and does it differ by cause? | Per-state distributions, averages and standard deviations; correlation between variables | Descriptive statistics; overlap between neighbouring states |
| 2. Engineering baseline | How well does conventional threshold monitoring perform? | Documented rules on rolling averages; thresholds derived from the training data's normal periods | Per-class precision, recall, F1; confusion matrix |
| 3. ML comparison | Do Logistic Regression and Random Forest add value over thresholds? | Raw metrics plus simple rolling features; class weighting; repeated over seeds | Per-class and macro F1, recall, confusion matrix, mean ± standard deviation across seeds |
| 4. Early warning | See section 8 | Episode-level evaluation | See section 8 |
| Robustness | Do detectors generalise to an unseen degradation cause? | Train on congestion episodes only; evaluate on link-quality episodes | As Experiments 3 and 4 |
| Failure cases | Where and why do detectors fail? | Inspection of false positives, false negatives and confused states | Error counts by state and cause; annotated examples |

**Why accuracy is not the primary metric.** Most minutes of a realistic link are
normal, so a detector that always answers "normal" would score high accuracy while
detecting nothing. Precision, recall and F1 per class show what accuracy hides.

## 8. Experiment 4: detection, early warning and prognosis

| Layer | Question | In scope | How measured |
|---|---|---|---|
| **Detection** | Is the link currently degrading? | Yes | Recall on degrading minutes (EARLY and DEGRADED) in all degrading episodes, including recovering ones; false alarms per 24 h of healthy operation |
| **Early warning** | How much warning is there before the critical-degradation threshold? | Yes | In episodes that enter SEVERE_DEGRADATION: lead time (minutes from the first alert after onset to threshold entry), and the share of episodes warned in advance |
| **Prognosis** | Will this degradation recover or worsen? | **No (future research)** | Not attempted |

**Alert classification rules (fixed before any detector existed):**

| Where the alert occurs | Counted as |
|---|---|
| Hidden severity exactly 0 (stable episodes, or before onset) | False alarm |
| Recovering episode while degrading | Correct degradation detection, reported separately as "warning that did not precede severe degradation". **Not** a false alarm |
| Worsening episode after onset | Detection; the earliest such alert defines the lead time |
| Severity between 0 and 0.10 (labelled NORMAL, but degradation has begun) | Excluded from false-alarm counts, because the ground truth is ambiguous |

Whether an episode entered the critical-degradation threshold is determined from its
labels, not from its planned episode type.

## 9. Research integrity commitments

1. Synthetic data is never presented as real measurement.
2. Generator parameters are frozen before experiments are run. They are not adjusted
   after seeing detector performance in order to obtain better-looking results. If a
   change becomes scientifically necessary (for example, a modelling error), the reason
   is explained before the change is made, and the change is recorded below.
3. Negative or weak results, including poor generalisation in the robustness check,
   are reported as results.
4. Results are reported with variation across seeds and without false precision.

## 10. Telemetry Integrity & Validation Layer ("Trust Layer")

> Before asking a model to learn from telemetry, can we establish that the
> observations themselves are structurally and operationally trustworthy?

The layer is deliberately small: plain rules in
[`netsense/data_validation.py`](../netsense/data_validation.py), run before any
detector. **A PASS status means only that no defined integrity or validation rule found
a problem. It does not certify that the telemetry is objectively correct.**

### 10.1 Field roles

| Role | Fields | A problem here means |
|---|---|---|
| Record keys | `episode_id`, `minute`, `timestamp` | FAIL: the observation cannot be placed in time or episode |
| Detector telemetry | the six observed metrics | FAIL: the analysis needs every value |
| Operational status | `connection_state` | WARN: not a detector input; status-based checks skip the minute |
| Study labels (synthetic only) | `episode_type`, `cause`, `hidden_severity`, `state` | FAIL, reported separately as study-label integrity. No telemetry rule uses label values |

### 10.2 Levels and rules

- **FAIL**: structurally impossible or record integrity broken. The row is quarantined.
- **WARN**: incomplete or operationally inconsistent but not impossible. Kept and reported.
- **INFO**: statistically unusual but potentially legitimate. Kept; never affects status.

| ID | Rule | Level |
|---|---|---|
| S1 | Schema: required column missing (FAIL); status column missing or unexpected column (WARN) | FAIL / WARN |
| S2 | Missing values, by field role | FAIL / WARN |
| S3 | Invalid category: status (WARN); study labels (FAIL) | WARN / FAIL |
| S4 | Negative telemetry value | FAIL |
| S5 | Percentage above 100 | FAIL |
| S6 | Throughput above the configured 100 Mbps capacity | FAIL |
| I1 | Duplicate episode/minute | FAIL |
| I2 | Duplicate timestamp within an episode | FAIL |
| I3 | Timestamp earlier than the previous record's (file order) | FAIL |
| C1 | Gap of more than 60 s between consecutive timestamps | WARN |
| O1 | `connection_state` disagrees with "UNSTABLE ⇔ loss ≥ 5%" | WARN |
| O2 | Latency **and** jitter identical across ≥ 3 consecutive minutes | WARN |
| O3 | Error rate = 100% while throughput > 0 | WARN |
| U1 | Value above Q3 + 3 × IQR of its column (the "far-out" rule) | INFO |

Rules use only knowledge an operator would have (link capacity, sampling interval,
status rule, allowed values), not quirks of the generator.

**Quarantine:** any FAIL row is removed; exact duplicates keep their first copy;
conflicting duplicates are all removed; WARN and INFO rows are kept; gaps are not filled.
The retained rows are called **validated telemetry**, not "clean" or "correct" telemetry.

### 10.3 Data-quality anomaly vs network anomaly

A *data-quality anomaly* means something may be wrong with the observation itself
(for example, loss = 140%). A *network anomaly* means the observation may be perfectly
valid but shows abnormal network behaviour (for example, loss = 20% with 80 ms latency).
The Trust Layer handles only the first. It must never remove legitimate severe
observations merely because they are statistically unusual; that is why U1 is INFO only.

### 10.4 Evaluation against injected defects

A deterministic function
([`netsense/defect_injection.py`](../netsense/defect_injection.py)) corrupts a **copy**
of the generated input with 13 defect types (25 events each). Every event is recorded
in a ground-truth manifest. Examples: missing values (poll timeout), duplicates
(collector retry), negative latency (measurement or timestamp-processing error), counter
wrap (throughput above capacity), outage gaps, out-of-sequence records, inconsistent
status and frozen pollers.

One type, `plausible_offset` (latency raised by 20%), is **undetectable by design**.
Rule-based validation can identify violations of known constraints, but it cannot
prove that plausible measurements are accurate.

Because the same author wrote the defects and the rules, near-complete detection
verifies the implementation; it is **not** evidence of real-world effectiveness. The
informative results are the flags on untouched rows, the missed `plausible_offset`, and
the outlier counterfactual. Outputs are in `results/metrics/` (validation reports, defect
manifest, detection table, untouched-row flags, outlier counterfactual), produced by
`python -m experiments.e0_telemetry_validation`.

### 10.5 Simulator simplifications exposed by validation

These are properties of the simulator, **not real-network findings**. They are kept
unchanged under the parameter freeze and will be discussed as limitations.

- **Saturated error counter (O3).** In the most severe link-quality minutes, counter
  noise and interference bursts push the reported error rate to 100% while traffic is
  still delivered. This is physically inconsistent. O3 flags these rows as WARN; they are
  kept, not quarantined.
- **Flat throughput under saturated congestion.** Once demand exceeds capacity,
  throughput is clipped to a near-constant value with no counter noise. For this reason
  the frozen-value rule (O2) is restricted to probe-based latency and jitter.

## 11. Experiment 1: network behaviour (method and pre-recorded expectations)

This section was written **before** Experiment 1 was run. Results are added below it only
after review, and are compared against these expectations without changing anything.

### 11.1 Data

- Pipeline: generate → Telemetry Integrity & Validation Layer → **validated telemetry**.
  The run stops if the validation status is FAIL.
- **Training episodes only.** The episode-level split
  ([`netsense/splits.py`](../netsense/splits.py)) is created here and then frozen and reused
  by Experiments 1–4. It is stratified by episode type and cause, takes exactly half of
  each group (40/40 stable, 30/30 worsening per cause, 15/15 recovering per cause), and
  uses its own random stream (seed + 2000). No test episode enters Experiment 1.
- Main tables and figures use seed 42. Headline values are repeated for all five seeds.
- In any analysis spanning more than one seed, an episode is identified by
  **(seed, episode_id)**, never by `episode_id` alone.

### 11.2 Analyses

1. **Distributions** of the six metrics by state and cause (Figure 2), plus a summary table
   (mean, standard deviation, median, 25th/75th percentiles and, for packet loss, the
   share of minutes with any loss; medians of loss are distorted by the 1/600 probe
   resolution).
2. **Separability** of EARLY_DEGRADATION and DEGRADED from NORMAL, per cause and metric:
   - *Outside-reference share*: % of minutes outside the **empirical NORMAL reference
     range**, the 1st–99th percentile of NORMAL minutes. If a state looked identical to
     NORMAL, about 2% would fall outside by chance. This range was fixed before running
     as a simple descriptive comparison. **It is not an engineering or operational alarm
     threshold and will not be presented as one.**
   - *Standardised difference*: (state mean − NORMAL mean) ÷ NORMAL standard deviation.
   - Each is computed **pooled** (NORMAL minutes of all links together) and
     **relative to each link's own baseline** (each value ÷ the median of that episode's
     first 60 minutes). The baseline view is possible only because, in this controlled
     simulation, every episode is healthy for at least its first 60 minutes. Real systems
     do not provide a perfectly labelled healthy first hour, so later detectors will use
     rolling recent-history baselines where appropriate.
3. **Spearman correlation** (correlation of ranks) between the six metrics, separately per
   cause (Figure 3). It is read as descriptive evidence of overlapping information between
   variables. It is not causal and is not, on its own, a reason to remove a feature.
4. **Rising phase: recovering vs worsening** (Figure 4). The question: *before the
   trajectories diverge, is there observable telemetry evidence of whether degradation
   will recover or continue towards severe degradation?*
   - Shared rising window: from onset to the first minute hidden severity reaches 0.25,
     the lowest possible recovery peak. Every episode of both types passes through this
     window before diverging. Hidden severity defines the window for analysis only.
   - One summary per episode: the baseline-relative **level** (median over the last 10
     minutes of the window) and **rise speed** (change in level per minute across the
     window), for each metric.
   - Comparison: **probability of superiority**, the chance that a randomly chosen
     worsening episode has a higher value than a randomly chosen recovering one
     (0.5 = no difference). Training episodes of all five seeds are pooled (75 recovering
     and 150 worsening per cause), with the range across seeds also reported.
   - No classifier is trained and no hypothesis test is used.

### 11.3 Expectations recorded before running

| Item | Expectation |
|---|---|
| Congestion, EARLY vs NORMAL | Latency, jitter and throughput rise modestly; large overlap in the pooled view; error rate unchanged |
| Congestion, DEGRADED vs NORMAL | Latency and jitter clearly separated; loss still mostly overlapping |
| Link quality, EARLY vs NORMAL | Error rate rises in relative terms, but the pooled overlap is large because absolute values stay tiny; latency and throughput nearly unchanged |
| Link quality, DEGRADED vs NORMAL | Error rate clearly separated; loss and retransmissions starting to separate; jitter rising slightly |
| Pooled vs baseline-relative | Separation is stronger in the baseline-relative view than in the pooled view |
| Packet loss | About half of NORMAL minutes contain at least one lost probe (background loss with 0.17% resolution), so loss is weak evidence until demand exceeds capacity or errors become large |
| Correlation, congestion | Latency, jitter and throughput strongly positively related; error rate largely unrelated to the others |
| Correlation, link quality | Error rate, loss and retransmissions strongly positively related; throughput negatively related to them |
| Rising phase | Probability of superiority close to 0.5 for every metric and both causes, because both episode types rise through the same process. A value far from 0.5 would suggest a simulator asymmetry, to be investigated and reported, not tuned |
| Across seeds | The same qualitative conclusions for all five seeds |

### 11.4 Observed results (added after the run; section 11.3 left unchanged)

Training episodes only: 650 across five seeds (130 per seed), each identified by
(seed, episode_id). No test episode entered the analysis. The validation status was WARN
for every seed, caused only by O3. Runtime was about 23 s. Outputs:
`results/metrics/e1_*.csv` and Figures 2–4 in `results/figures/`.

**Implementation decisions made before any result existed:**
- A ratio to a baseline of 0 is undefined. Baseline-relative packet loss therefore exists only
  for episodes with non-zero first-hour median loss (34% of NORMAL minutes in seed 42).
- The NORMAL reference uses every NORMAL minute, stable episodes included.
- Episodes that never reach severity 0.25 are excluded from the rising-phase comparison.
  None were.

**Separability from NORMAL.** Shown as % of minutes outside the empirical NORMAL reference
range, pooled view, minimum–maximum over five seeds. About 2% is expected by chance.

| Cause | Metric | EARLY_DEGRADATION | DEGRADED |
|---|---|---|---|
| Congestion | Latency | 4.6–7.6 | 40.3–47.7 |
| Congestion | Jitter | 4.6–7.1 | 41.8–47.0 |
| Congestion | Packet loss | 3.0–3.8 | 12.1–16.4 |
| Congestion | Retransmissions | 4.0–5.9 | 13.8–18.2 |
| Congestion | Throughput | 4.5–7.2 | 42.6–47.4 |
| Congestion | Error rate | 1.4–3.3 | 1.6–2.9 |
| Link quality | Error rate | 4.7–18.1 | 83.6–98.0 |
| Link quality | Retransmissions | 3.8–12.4 | 77.2–94.3 |
| Link quality | Packet loss | 3.3–4.2 | 46.3–54.2 |
| Link quality | Latency / jitter / throughput | 1.6–6.0 | 1.1–5.4 |

The baseline-relative view gave similar values, with differences in both directions
(seed 42, DEGRADED: congestion jitter 50.2 vs 44.2 pooled; congestion throughput 34.4 vs
44.6; link-quality loss 44.5 vs 51.2).

**Packet loss.** Minutes with at least one lost probe: NORMAL 47–48% for both causes;
EARLY 51% (congestion) and 63% (link quality); DEGRADED 55% and 94%; SEVERE 99.5% and 100%.

**Spearman correlation (seed 42).**
- Congestion: jitter–throughput 0.95, latency–throughput 0.82, latency–jitter 0.81,
  loss–retransmissions 0.74; error rate ≤ 0.08 with every other metric.
- Link quality: error rate–retransmissions 0.90, loss with error rate and with
  retransmissions 0.84, jitter with these 0.68–0.69; throughput −0.29 to −0.30.

**Rising phase.** Probability of superiority across 24 comparisons (six metrics × two
measures × two causes) ranged from 0.43 to 0.59 with five seeds pooled. For **every**
comparison the per-seed range spans 0.5, so no direction is consistent between seeds.
Values at least 0.05 from 0.5:
- congestion latency level 0.43; congestion throughput speed 0.55;
- link-quality latency level 0.56, jitter level 0.58, loss level 0.59 (on 61 vs 31 episodes
  where the loss ratio is defined), retransmissions level and speed 0.55.

No significance tests were used. By the end of the rising window, degradation itself is
visible: for example, congestion jitter is about 2× and link error rate about 5.8× the
episode's own baseline.

### 11.5 Expectations compared with observations

| Pre-recorded expectation (11.3) | Observed | Verdict |
|---|---|---|
| Congestion EARLY: modest rise in latency, jitter, throughput; large pooled overlap; error rate unchanged | 5–8% outside reference; error rate at chance | Matches |
| Congestion DEGRADED: latency and jitter clearly separated; loss mostly overlapping | Latency and jitter about 40–56% outside; loss 11–20% | Partly: about half still overlaps |
| Link EARLY: error rate rises relatively; large pooled overlap; latency and throughput unchanged | Median error rate about 7× NORMAL, yet 82–95% of minutes inside the reference range | Matches |
| Link DEGRADED: error rate clear; loss and retransmissions starting to separate; jitter slightly up | Error rate 84–99%; retransmissions 77–95%; loss 34–54%; jitter 1.2–1.6% | Partly: retransmissions stronger than expected; jitter does not separate |
| Baseline view separates better than pooled | Small differences in both directions | Does not match |
| About half of NORMAL minutes contain loss | 47–48% | Matches |
| Congestion correlations | As expected | Matches |
| Link correlations: throughput negatively related | Weakly (−0.29 to −0.30) | Partly |
| Rising phase close to 0.5 | 0.43–0.59; every per-seed range spans 0.5 | Matches |
| Same conclusions across seeds | Yes; large seed spread for link EARLY error rate (4.7–18.1%) | Matches |

The mismatches are kept as findings. The expectations were not rewritten.

### 11.6 Interpretation

All statements describe the frozen synthetic environment, not real networks.

1. **EARLY degradation overlaps strongly with NORMAL at individual-minute level.** At most
   about 18% of early-degradation minutes (link-quality error rate) and about 8% (congestion
   latency and jitter) fall outside the empirical NORMAL reference range.
2. **Per-link baseline normalisation did not materially improve separation and sometimes
   reduced it.** Overlap seems to be driven mainly by minute-to-minute variation (traffic
   variation, harmless bursts, measurement noise) rather than by differences between links.
3. **Hypothesis for later experiments, not an established finding:** within-link temporal
   variation and persistence (for example rolling windows) may matter more than static
   per-link calibration.
4. **Recovering and worsening episodes show no consistent separation during their shared
   rising phase.** This supports *degradation detection ≠ prognosis*. Prognosis remains
   future research.
5. **At 600 probes per minute, low packet-loss values are poor evidence of early
   degradation.** About half of healthy minutes already lose at least one probe, and the
   loss medians (0 or 0.17%) are a resolution artefact.
6. **Correlation across the full operating range is not evidence that a variable is useful
   for early detection.** Example: under link quality, jitter correlates 0.68 with error rate,
   yet DEGRADED jitter is barely separable from NORMAL. The correlation comes mostly from
   severe minutes.
7. **Congestion and link-quality degradation show different telemetry relationships.**
   Under congestion, delay, jitter and throughput move together and error rate is
   unrelated. Under link quality, error rate, loss and retransmissions move together and
   delay barely changes.

Explanations of the mismatches (not changes): harmless traffic bursts widen the NORMAL jitter
range, so the slight rise in link-quality DEGRADED jitter stays inside it. Retransmissions
come from counters and avoid the 1/600 probe resolution that limits loss. The DEGRADED band
for congestion includes utilisation that is not yet far outside the burst-widened normal
range. Differences in normal load between links dominate throughput until link-quality
degradation becomes severe.

## 12. Experiment 2: engineering monitoring baseline (method and pre-recorded expectations)

This section was written **before** any Experiment 2 detector was run. Results are added
later in separate subsections; this text is not rewritten afterwards.

### 12.1 Question

> Can a transparent persistence-aware engineering detector improve degradation detection
> and false-alarm control compared with single-minute threshold rules?

This is a non-ML baseline. Its frozen specification is the benchmark that Experiment 3 must
be compared against, on the same held-out episodes and with the same event-level metrics.
It cannot show how the rules behave on real telemetry, that any persistence setting is
optimal, or anything about prognosis.

### 12.2 Detectors

**Per-minute condition (cause-agnostic):** a minute is *suspicious* if **any** of latency,
jitter, error rate or retransmission rate is strictly above its threshold.

- Probe packet loss is excluded: about half of healthy minutes lose at least one of 600
  probes, so low values are a resolution artefact (Experiment 1). The retransmission
  counter measures the same phenomenon without that limit.
- Throughput is excluded: high utilisation is not in itself a fault, and harmless bursts
  drive it.

**Thresholds:** the **99th percentile of NORMAL minutes in the training episodes of all five
seeds pooled**, one value per indicator, upper side only. This is a pre-declared empirical
engineering threshold derived from healthy training telemetry. **It is not an established
telecommunications alarm standard.** Thresholds are absolute rather than baseline-relative,
because per-link normalisation did not help in Experiment 1. Per-seed values are reported
for stability only and are never used.

**Persistence rules**, all expressed as "at least K of the last W minutes suspicious":

| Detector | K of W | Meaning |
|---|---|---|
| A | 1 of 1 | Single minute (reactive reference) |
| B | 3 of 3 | 3 consecutive minutes |
| C1 | 3 of 10 | 3 of the last 10 minutes; gaps tolerated |
| C2 | 6 of 20 | 6 of the last 20 minutes; same density as C1, longer memory |

A → B isolates persistence, B → C1 isolates gap tolerance (K = 3 in both), and C1 → C2
isolates window length (30% density in both). Only data up to the current minute is used.
Counters reset at the start of every episode.

**Post-detection diagnostic signature (reporting only):** the indicator group suspicious at
the start minute of an episode's first valid detection. *Delay* means latency or jitter;
*error* means error rate or retransmission rate.

### 12.3 Alarm events (identical for all detectors)

- An alarm **starts** at the first minute the detector's rule is satisfied while no alarm is
  active.
- It stays active while the rule is satisfied. If the rule becomes satisfied again before
  closure is confirmed, the event continues; there is no new event.
- **Causal clearing:** closure is **confirmed** only on the 5th consecutive minute in which
  the rule is not satisfied. The monitor never uses future observations to close earlier.
  Two times are recorded:
  - *closure confirmation time*: the minute closure was confirmed online;
  - *effective event end*: the last minute in which the rule was satisfied, a
    retrospective description.
- One continuous alarm is **one event**. An episode may have several events. An event still
  open at the end of an episode is closed there (no confirmation time).
- The alarm is shown **online** from its start to its confirmation time (or episode end).
  Alarm burden and "alarm already active at onset" use this online span.

### 12.4 Ground truth and metrics (evaluation only)

Hidden labels never enter the detector.

**Terms**
- *Onset* is the generator's recorded `onset_minute`.
- *Severe entry* is the first minute with hidden severity ≥ 0.70.
- *Degrading phase:* from onset to episode end (worsening episodes), or to the last minute
  with severity above 0 (recovering episodes).
- *Healthy period:* every minute outside a degrading phase (all of every stable episode,
  the pre-onset part of every degrading episode, and the post-recovery part of recovering
  episodes).

**Implementation clarification, decided before running:** inside a degrading phase,
severity can be clipped to exactly 0 for a moment by the random wobble. Such minutes belong
to the degrading phase, not to healthy time. This implements the existing rule
(section 8) that alarms after onset are detections and are not counted as false alarms
while ground truth is ambiguous.

**Alarm events are classified by their start minute:**
- start in a healthy period → **false alarm**;
- start inside a degrading phase → **valid detection**.

An alarm that began before onset is a false alarm and never receives early-detection
credit.

**Episode outcomes (degrading episodes):**
1. *valid post-onset detection*: an event starts inside the degrading phase;
2. *alarm already active at onset*: no such event, but an alarm that started before onset
   was still active (online) at onset;
3. *no valid degradation alarm*.

The share of all degrading episodes with an alarm active at onset is also reported as
context. Detection rate and delay use outcome 1 only.

**Metrics:**
- detection rate (worsening, recovering), with the three outcome shares;
- false alarms per 100 healthy hours;
- healthy alarm burden (% of healthy minutes under an online alarm);
- detection delay (first valid detection − onset; median and 25th–75th percentile);
- % of worsening episodes warned before severe entry;
- severe lead time (severe entry − first valid detection; median and 25th–75th percentile;
  negative values kept).

### 12.5 Protocol

- **Phase 1 (freeze), training episodes only:** test episodes are dropped immediately
  after generation, using only episode metadata (id, type, cause), before any telemetry
  value is validated, summarised or used. Then: compute thresholds, run all four detectors
  on training episodes, and write the frozen specification
  `results/metrics/e2_detector_spec.json`. The specification is committed before Phase 2.
- **Phase 2 (evaluate), held-out test episodes, once:** load the committed specification
  without recomputing anything. Any change after Phase 2 becomes a separately named new
  experiment.
- Episodes are identified by (seed, episode_id). Results are reported pooled and per seed.
- **Burst attribution** of false alarms was planned only if it needed no change to the
  frozen generator. The generator does not store burst timing, and recovering it would mean
  intercepting the generator's internal functions at run time. That is treated as a
  modification, so burst attribution is **not performed**.
- **Terminology:** "error rate" refers to the `error_rate_pct` telemetry column.

### 12.6 Expectations recorded before running

| Item | Expectation |
|---|---|
| A (single minute) | Highest false-alarm rate (four indicators at about 1% each, plus bursts); shortest delay; highest detection rates |
| B (3 consecutive) | Clearly fewer false alarms than A, but harmless bursts of 3–10 minutes still trigger it; noticeably later detection; often misses EARLY degradation because exceedances are scattered, so detection happens mostly in DEGRADED |
| C1 (3 of 10) | False alarms between A and B; earlier detection than B because gaps are tolerated |
| C2 (6 of 20) | Fewest false alarms; longest delay |
| Worsening episodes | Close to 100% detected by every detector eventually; differences lie in delay |
| Recovering episodes | Lower detection rate than worsening, especially low-peak episodes, and lowest for C2 |
| Cause | Link-quality degradation detected earlier than congestion |
| Before severe | Most worsening episodes warned before severe entry; lead time shrinks as persistence increases |
| Overall | No detector achieves both low false alarms and early detection; persistence trades false alarms for delay, by an amount not known in advance |
| Diagnostic signature | Congestion first flagged by delay indicators; link quality by error indicators |
| Training vs test | Similar, because no performance-based tuning is done |

### 12.7 Phase 1: training results and frozen specification

*Added after Phase 1. Sections 12.1–12.6, including the pre-recorded expectations in 12.6,
are unchanged. These are **training results only** and provide **no evidence yet about
held-out generalisation**.*

**Frozen specification:** [`results/metrics/e2_detector_spec.json`](../results/metrics/e2_detector_spec.json).
The thresholds are the 99th percentile of 141,174 NORMAL training minutes (650 training
episodes, five seeds pooled):

| Indicator | Threshold | Per-seed range (stability only, not used) |
|---|---|---|
| Latency | 47.73 ms | 46.25–50.30 |
| Jitter | 24.31 ms | 22.58–26.43 |
| Error rate | 0.3027% | 0.2295–0.3864 |
| Retransmission rate | 0.2828% | 0.2435–0.3240 |

2.13% of healthy training minutes are suspicious under the four-indicator OR. Healthy
suspicious minutes come in runs: 46% of runs last 1 minute, 20% last 2, 12% last 3, and 22%
last 4 minutes or more.

**Training results** (pooled; `results/metrics/e2_results_train*.csv`):

| Metric | A (1 of 1) | B (3 of 3) | C1 (3 of 10) | C2 (6 of 20) |
|---|---|---|---|---|
| Detection rate, worsening | 100% | 100% | 100% | 100% |
| Detection rate, recovering | 88.0% | 78.0% | 80.7% | 72.7% |
| False alarms per 100 healthy h | 42.1 | 17.8 | 21.3 | 4.9 |
| Healthy alarm burden | 5.9% | 2.2% | 5.0% | 1.6% |
| Median delay, min (25th–75th) | 36 (23–50) | 46 (33–62) | 43 (29–59) | 51 (39–66) |
| Warned before severe | 100% | 99.3% | 100% | 98.0% |
| Median severe lead, min (25th–75th) | 40.5 (28–60) | 31 (17–48) | 34 (22–52) | 26 (16–41) |
| Alarm active at onset (any) | 4.0% | 1.3% | 4.9% | 0.9% |

**Phase 1 findings (training data):**
- All four detectors eventually detected 100% of worsening training episodes.
- Persistence substantially reduced false-alarm events but increased detection delay.
- B substantially reduced false alarms relative to A (−58%) at about 10 minutes more median
  delay.
- C1 gained only modest detection speed over B (3 minutes) while increasing false alarms and
  healthy alarm burden. Once triggered, its 10-minute memory keeps alarms on longer.
- C2 produced the lowest false-alarm rate but the longest delay and the lowest
  recovering-episode detection.
- Persistence did not eliminate false alarms. Healthy suspicious observations occur in
  clusters and runs (harmless bursts), not only as isolated minutes.
- Recovering degradation was harder to detect than worsening degradation.
- Link-quality degradation was detected earlier than congestion, and congestion gave
  shorter severe lead times under persistence.
- Diagnostic first-trigger signatures broadly matched the simulated mechanisms:
  link-quality alarms began from error indicators in 89–94% of detected episodes, and
  congestion alarms mostly from delay indicators, often together with others.
- **Detector A's apparent early advantage must be interpreted cautiously.** About 12% of its
  first valid detections in worsening episodes occurred while severity was still below 0.10,
  shortly after onset. Some of these may be coincidental healthy-like exceedances that the
  pre-declared rule credits as detections.
- Alarm-already-active-at-onset cases were rare: 0.9–4.9% of degrading episodes, and only
  one episode (detector A) without a later valid detection.
- Burst attribution was not performed because it would have required modifying the
  frozen generator.

**Interpretation.** No detector is declared universally "best". The results describe an
engineering trade-off between four things: false-alarm suppression, detection delay,
sensitivity to recovering degradation, and severe-degradation lead time. Whether this
trade-off holds on held-out episodes is the question for Phase 2.

### 12.8 Phase 2: held-out evaluation

*Added after the single held-out evaluation. Sections 12.1–12.7 are unchanged.*

- **Pre-test specification.** Commit `ddd57ca` was the immutable pre-test detector
  specification. The evaluation refused to run unless the detector, evaluation, config,
  generator, split and specification files were byte-identical to that commit.
- **Thresholds** were loaded from the committed `e2_detector_spec.json` and were not
  recomputed.
- **Scope.** 650 held-out episodes were evaluated (300 worsening, 150 recovering, 200
  stable; 2,227 healthy hours; five seeds). Zero training episodes entered Phase 2. Episodes
  were identified by (seed, episode_id).
- **Integrity.** No Phase 2 software bug occurred, and no analytical tuning was performed
  after test exposure. Figure layout was finalised on a training-data dry run before the
  test run. The only change afterwards was presentation (label placement; Figure 5's delay
  axis starting at 0), re-rendered from the saved result files without re-evaluating.

**Held-out results** (pooled; `results/metrics/e2_results_test*.csv`, training comparison in
`e2_train_vs_test.csv`):

| Metric (test; change vs training) | A (1 of 1) | B (3 of 3) | C1 (3 of 10) | C2 (6 of 20) |
|---|---|---|---|---|
| Detection rate, worsening | 100% (0) | 100% (0) | 100% (0) | 100% (0) |
| Detection rate, recovering | 90.7% (+2.7) | 77.3% (−0.7) | 82.0% (+1.3) | 73.3% (+0.7) |
| False alarms per 100 healthy h | 43.4 (+1.3) | 18.0 (+0.2) | 21.5 (+0.1) | 6.4 (+1.5) |
| Healthy alarm burden | 6.1% (+0.2) | 2.3% (+0.1) | 5.2% (+0.2) | 2.1% (+0.4) |
| Median delay, min (25th–75th) | 36.5 (23–51) (+0.5) | 47 (32–64) (+1.0) | 45 (31–60) (+2.0) | 55.5 (41–71) (+4.5) |
| Warned before severe | 100% (0) | 99.3% (0) | 100% (0) | 99.3% (+1.3) |
| Median severe lead, min | 42 (+1.5) | 32 (+1.0) | 34 (0) | 26 (0) |
| Valid / active at onset only / no valid alarm (of 450) | 436 / 0 / 14 | 416 / 1 / 33 | 423 / 0 / 27 | 410 / 2 / 38 |

**Findings on held-out data**
- All four detectors detected 100% of worsening test episodes.
- A had the shortest median delay but the highest false-alarm rate and burden.
- C2 had the lowest false-alarm rate but the longest delay and the lowest recovering-episode
  detection.
- B occupied an intermediate trade-off.
- C1 produced only a modest timing improvement over B (2 minutes) while increasing false
  alarms (21.5 vs 18.0 per 100 h) and healthy alarm burden (5.2% vs 2.3%).
- Persistence therefore suppressed false alarms at the cost of detection speed and
  sensitivity to recovering degradation, and severe-degradation lead time decreased as
  persistence increased (42 → 32/34 → 26 minutes).
- Link-quality degradation was detected earlier than congestion, by 8–17 minutes in median
  delay.
- First-trigger signatures: link-quality detections began from error-type evidence in
  87–91% of cases. Congestion detections began from delay-type evidence alone in 54–67%,
  from both groups in 16–35%, and from error-type evidence alone in 3–17%.
- **Training and held-out results were broadly consistent, with a small worse-direction tilt
  on test:** every false-alarm and delay change was zero or slightly unfavourable. This is
  held-out generalisation within the same frozen simulator family, **not evidence of
  real-network generalisation.**

**Unexpected results, retained as observed**
- C2's test false-alarm rate was worse than training (6.4 vs 4.9 per 100 h).
- C2's test median delay was worse than training (55.5 vs 51 minutes), above every
  training seed's value (49–54), although the test per-seed values (51–57) overlap that range.
- **Recovering detection by cause reversed:** on test, A detected 94.7% of recovering
  congestion episodes vs 86.7% of link-quality ones, whereas training showed 81% vs 95%
  (75 episodes per cause). This is retained as an observed subgroup result.
- Alarm-already-active-at-onset cases (with no later valid detection) occurred occasionally:
  B once, C2 twice (none in training).
- **The onset-credit caveat remains.** 12.0% of A's first valid detections in worsening
  episodes occurred while hidden severity was still below 0.10 (training: 12.3%); for B,
  C1 and C2 the figures were 6.3%, 7.7% and 1.7%. Some very-early post-onset detections
  may be coincidental threshold excursions. In the Figure 6 link-quality example, A, B and
  C1 first fired on a short error-rate spike during early degradation, before the
  sustained threshold crossing. None of these detections were reclassified.

**Figure 6 episode selection, stated operationally:** for each cause, the seed-42 held-out
worsening episode whose onset-to-severe duration is closest to the sample median
onset-to-severe duration was selected; if several episodes were equally close, the lowest
`episode_id` was chosen. With an even number of episodes (30 per cause) the sample median
may lie between observed durations. This clarifies the existing deterministic rule; it is
not a new selection. Selected: congestion (42, 18), duration 79 min against a median of 81
(tied with episode 116 at 83 min); link quality (42, 37), duration 88 min, equal to the
median (tied with episode 144). The figure caption's phrase "the median onset-to-severe
duration" refers to this rule.

### 12.9 Interpretation and limitations

All findings describe the frozen synthetic environment.

1. **Persistence works, but not for free.** Temporal persistence substantially reduced
   false-alarm events but increased detection delay.
2. **No universally best detector exists.** Choosing a detector is an engineering trade-off
   among false alarms, delay, sensitivity to recovering degradation and severe-degradation
   lead time.
3. **100% eventual worsening detection is insufficient as a performance claim.** All
   detectors eventually detected every worsening episode, but they differed substantially
   in *when* they did so.
4. **Short gap-tolerant persistence added limited value.** C1 detected only modestly earlier
   than B while increasing false alarms and healthy alarm burden.
5. **Healthy abnormal observations cluster.** Persistence removes isolated noise more
   effectively than sustained harmless variation and bursts.
6. **Mechanism signatures differ within the simulator.** Congestion was predominantly
   associated with delay-type first evidence and link-quality degradation with error-type
   evidence. This reflects the simulated mechanisms and is not a real-network causal finding.
7. **Held-out consistency is limited evidence.** Agreement between training and test supports
   reproducibility within the simulator but does not establish external validity.

**Limitations specific to Experiment 2**
- The threshold rule (99th percentile), the persistence values and the 5-minute clear-delay
  shape the absolute numbers; the comparisons between detectors are the focus.
- False-alarm rates depend on the simulated burst model.
- Burst attribution was not possible without modifying the generator.
- Onset is a modelling convention: delays are measured from a point where the physical
  signal is still negligible.
- There is no cost model weighing missed detections against false alarms.
- Only one held-out split was evaluated.

## 13. Experiment 3: interpretable ML detector (method and pre-recorded expectations)

This section was written **before** any Experiment 3 model was fitted. Results are added
later in separate subsections; this text is not rewritten afterwards.

### 13.1 Question

> Can an interpretable ML detector use combinations and recent temporal patterns in
> observable telemetry to improve the false-alarm vs detection-delay trade-off beyond the
> frozen persistence-aware engineering baseline?

The primary comparison is against the frozen Experiment 2 detectors, using the same
event-level metrics. Classification scores are secondary diagnostics.

### 13.2 Inputs and target

**Inputs.** The same four indicators as Experiment 2: latency, jitter, error rate and
retransmission rate. Probe packet loss and throughput are excluded, so that any difference
from Experiment 2 comes from how the signals are combined over time, not from extra
information.

**Fixed transform:** error rate and retransmission rate become `log10(value + 0.001)`.
Latency and jitter keep their natural scale.

**Twelve causal features** (3 per indicator), computed within each episode, keyed by
(seed, episode_id), on the full minute index:

| Feature | Definition at minute t |
|---|---|
| Current | x(t) |
| 10-minute mean | mean of x(t−9 … t) |
| 5-minute trend | mean(x(t−4 … t)) − mean(x(t−9 … t−5)) |

- **Causality:** only minutes up to *t* are used.
- **Gaps:** if any of the 10 minutes t−9 … t is missing for any indicator, all 12 features
  at *t* are undefined. Nothing bridges a gap.
- **Warm-up:** minutes 0–8 of each episode are therefore undefined.
- **Undefined features:** the minute is excluded from fitting and treated as *not
  suspicious* when alarms are produced.

**Target (training labels only; never a feature):**
- **0** = healthy minute, outside the degrading phase (section 12.4);
- **1** = inside the degrading phase with hidden state EARLY_DEGRADATION, DEGRADED or
  SEVERE_DEGRADATION;
- **excluded** = inside the degrading phase with hidden severity below 0.10 (labelled
  NORMAL), because the ground truth is ambiguous.

This is a *detection* target: it describes the link at minute *t* only, never the episode's
future outcome.

### 13.3 Model, preprocessing and score

- scikit-learn `LogisticRegression` with library defaults (L2 penalty, C = 1.0, lbfgs
  solver), `max_iter = 1000`. No class weighting, no hyperparameter search, no feature
  selection, no comparator model.
- Features are standardised with the mean and standard deviation (population form) of the
  **training fitting rows only**.
- The output is called the **logistic model score**, or the predicted probability under the
  fitted logistic model. **Probability calibration is not performed;** the score is not a
  calibrated probability of degradation.
- The fitted scaler, coefficients and intercept are stored as numbers in
  `e3_ml_spec.json`. Scores are reconstructed from that file, without refitting.

### 13.4 Threshold τ (training-only, deterministic)

**Target rate:** the share of healthy training minutes that the frozen engineering
per-minute condition flags: 2,834 of 133,361 = 2.125% (Experiment 2, Phase 1).

**Construction:**
1. Take the scores of all healthy training fitting rows.
2. For each distinct score value *s*, compute the share of those rows with score > *s*.
3. τ is the *s* whose share is closest to the target rate. If two values are equally
   close, choose the larger *s* (fewer alarms).
4. A minute is suspicious when **score > τ** (strict, matching Experiment 2).

Because ties in scores are possible, exact equality with the target is not claimed. The
achieved rate is reported. No alternative threshold is evaluated against detection
performance.

### 13.5 Persistence and alarms

The suspicious-minute sequence is passed unchanged through Experiment 2's frozen
`rule_satisfied` and `alarm_events` functions, with the same four rules (ML-A 1 of 1,
ML-B 3 of 3, ML-C1 3 of 10, ML-C2 6 of 20) and the 5-minute causal clear-delay. It is then
evaluated with the frozen `evaluation.py`.

Each ML variant is compared descriptively with its engineering counterpart. The ML features
already contain 10-minute smoothing, so ML plus persistence smooths twice.

### 13.6 Interpretability and diagnostics

- **Coefficients:** standardised coefficients, their signs, and odds ratios per standard
  deviation, reported descriptively and **not causally**. Current value, 10-minute mean
  and trend of the same indicator are correlated, so coefficient magnitudes and signs can
  redistribute across related predictors.
- **Leave-one-indicator-group-out sensitivity diagnostic:** for each indicator, the model is
  refitted without that indicator's three features, and the change in training PR-AUC is
  reported. This is a training sensitivity diagnostic, not proof of independent feature
  importance, and it never changes the final model.
- **Per-minute diagnostics:** PR-AUC, ROC-AUC, precision, recall and F1 at τ, and the share
  of healthy, EARLY, DEGRADED and SEVERE minutes flagged by the ML condition and by the
  frozen engineering condition, on the same rows.

### 13.7 Protocol

- **Phase 1:** training episodes only (test episodes dropped after generation using episode
  metadata only). Fit, derive τ, run the four ML variants on training episodes, write
  `e3_ml_spec.json`, then review and commit.
- **Phase 2:** load the committed specification and evaluate the held-out episodes once, with
  no refitting or tuning. A genuine bug found after test exposure means stopping and
  reporting it, not a silent rerun.

### 13.8 Expectations recorded before running

| Item | Expectation |
|---|---|
| Per-minute EARLY hit rate at the matched healthy rate | ML higher than the engineering condition, because 10-minute means average out the noise that hides small shifts |
| ML-A vs Eng-A | ML-A has far fewer false-alarm events (its signal is already smoothed), with similar or slightly longer delay |
| ML-B/C1/C2 vs Eng-B/C1/C2 | Smaller gains; engineering persistence already captures much of what the rolling features add, and may be hard to beat |
| Trade-off frontier | At least one ML variant at or slightly inside the engineering frontier, but this is not certain |
| Recovering episodes | Still harder to detect than worsening episodes |
| Congestion | Still detected later than link-quality degradation |
| Coefficients | Error-rate and retransmission features large and positive; jitter and latency 10-minute means positive; some signs within a correlated group may be counter-intuitive |
| Harmless bursts | Still cause some ML false alarms; 10-minute means dilute short bursts but not long ones |
| Training vs test | Similar |

If ML does no better than the engineering rules, or worse, that is reported as the finding.

### 13.9 Phase 1: training results

*Added after Phase 1. Sections 13.1–13.8, including the pre-recorded expectations in 13.8,
are unchanged.* **These are training (in-sample) results.** The model's 13 parameters were
fitted to these same training episodes, whereas the engineering thresholds were derived from
healthy training minutes only, without degradation labels. The comparison therefore favours
ML and **does not establish held-out superiority.** None of the observations below caused any
change to the model, features, transforms, windows, labels, threshold or persistence rules.

**Data.**
- 650 training episodes (234,000 rows).
- 220,337 fitting rows: 127,511 healthy and 92,826 positive.
- Excluded: 5,850 warm-up rows (minutes 0–8) and 7,813 post-onset rows with severity below
  0.10.
- No rows were excluded for gaps.

**Threshold.**
- Target healthy trigger rate: 2,834 / 133,361 = 2.1251%.
- τ = 0.6447583878236622 (logistic model score; not a calibrated probability).
- Achieved healthy trigger rate: 2.1253%.

**Per-minute diagnostics** (training fitting rows; `e3_minute_diagnostics_train.csv`):

| Metric | ML (score > τ) | Frozen engineering condition |
|---|---|---|
| PR-AUC / ROC-AUC | 0.964 / 0.962 | — |
| Precision / recall / F1 | 0.966 / 0.821 / 0.888 | 0.962 / 0.736 / 0.834 |
| Healthy trigger rate | 2.125% | 2.136% (same rows) |
| EARLY_DEGRADATION hit rate | 34.7% | 10.0% |
| DEGRADED hit rate | 87.6% | 72.1% |
| SEVERE_DEGRADATION hit rate | 100% | 99.98% |

**Event-level results** (training, pooled; `e3_results_train.csv`; engineering values from the
committed `e2_results_train.csv`):

| Metric | ML-A | Eng-A | ML-B | Eng-B | ML-C1 | Eng-C1 | ML-C2 | Eng-C2 |
|---|---|---|---|---|---|---|---|---|
| Detection, worsening | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| Detection, recovering | 89.3% | 88.0% | 86.7% | 78.0% | 87.3% | 80.7% | 86.0% | 72.7% |
| False alarms / 100 h | 24.2 | 42.1 | 11.5 | 17.8 | 12.1 | 21.3 | 8.6 | 4.9 |
| Healthy alarm burden | 4.2% | 5.9% | 2.2% | 2.2% | 3.8% | 5.0% | 3.6% | 1.6% |
| Median delay, min (25th–75th) | 32 (21–43) | 36 (23–50) | 37 (24–49) | 46 (33–62) | 36 (24–48) | 43 (29–59) | 40 (27–52) | 51 (39–66) |
| Warned before severe | 100% | 100% | 100% | 99.3% | 100% | 100% | 98.7% | 98.0% |
| Median severe lead, min (25th–75th) | 46 (32–64) | 40.5 (28–60) | 40.5 (26–60) | 31 (17–48) | 41 (27–61) | 34 (22–52) | 37 (22–57) | 26 (16–41) |
| No valid degradation alarm | 3.6% | 3.8% | 4.4% | 7.3% | 4.2% | 6.4% | 4.7% | 9.1% |

No ML variant had an "alarm already active at onset" outcome.

**Per-seed ranges (ML, training):**

| Detector | False alarms / 100 h | Median delay (min) | Recovering detection |
|---|---|---|---|
| ML-A | 22.4–29.2 | 29–33 | 83–93% |
| ML-B | 8.5–15.2 | 36–37 | 77–93% |
| ML-C1 | 9.6–15.4 | 35–37 | 80–93% |
| ML-C2 | 6.0–11.4 | 39–41 | 77–93% |

**Findings (training only):**
- **ML-A, ML-B and ML-C1 outperformed their engineering counterparts on both training false
  alarms and delay.**
- **ML-C2 was faster but noisier than Eng-C2:** 11 minutes earlier, but 8.6 vs 4.9 false
  alarms per 100 h and 3.6% vs 1.6% healthy burden.
- At an almost equal healthy trigger rate, ML flagged about 3.5 times as many EARLY minutes
  as the engineering condition.
- **Recovering degradation remained harder** to detect than worsening degradation (86–89%
  vs 100%).
- **Congestion remained later than link quality:** median delay 40–49 vs 27–33 minutes.
  Link-quality recovering episodes were all detected (100%); congestion recovering
  episodes 72–79%.

**Coefficients** (standardised; descriptive, not causal; `e3_coefficients.csv`, Figure 8):
- The largest are error-rate 10-minute mean (+4.34; odds ratio per SD 77), jitter
  10-minute mean (+2.40) and jitter current (+1.39).
- **All three latency coefficients are negative** (current −0.63, mean −0.07, trend −0.16).
- **The retransmission 10-minute mean coefficient is negative** (−0.27).
- The current value, 10-minute mean and trend of the same indicator are correlated, so
  coefficient magnitudes and signs can redistribute across related predictors. These signs
  do not describe physical effects.

**Leave-one-indicator-group-out sensitivity diagnostic** (change in training PR-AUC from
0.9637): latency −0.00042, jitter −0.0190, error rate −0.0065, retransmission −0.0000041.
Removing the latency or retransmission feature groups produced negligible change in training
PR-AUC, suggesting substantial redundancy with the remaining features within this fitted
simulator model. This is a training-only sensitivity diagnostic and does not establish
independent feature importance.

**Comparison with the pre-recorded expectations (13.8):**

| Expectation | Training result | Verdict |
|---|---|---|
| Higher EARLY hit rate than engineering | 34.7% vs 10.0% | Matches, more strongly than expected |
| ML-A far fewer false alarms, similar or slightly longer delay | −43% false alarms, 4 minutes *shorter* delay | Partly: better than expected on delay |
| ML-B/C1/C2 smaller gains; persistence hard to beat | B and C1 large gains in false alarms and delay; C2 faster but noisier than Eng-C2 | Does not match |
| At least one ML variant at or near the frontier | Three variants clearly inside it on training | Matches, more strongly than expected |
| Recovering still harder | Yes | Matches |
| Congestion later | Yes | Matches |
| Error and retransmission positive; jitter and latency means positive | Error-rate mean and jitter positive; retransmission mean and all latency coefficients negative | Partly |
| Bursts still cause false alarms | Yes | Matches |

**Surprises:**
1. The improvement over the engineering detectors was larger than the pre-recorded
   expectations.
2. ML-C2 produced more false alarms than Eng-C2. A possible but unverified explanation:
   the 10-minute mean keeps the score above τ for several minutes after a burst.
3. The latency coefficients are negative. This is unexplained; it may reflect redistribution
   among correlated predictors or differences in baseline delay between links. It is not
   verified and not causal.
4. The latency and retransmission groups were nearly redundant in the sensitivity diagnostic.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-08 | Initial plan | — |
| 2026-10-08 | Renamed `SEVERE_FAILURE` to `SEVERE_DEGRADATION`; defined entry into it as the critical-degradation threshold; reworded the primary question accordingly | "Failure" overstated the condition: the simulated link may still carry traffic |
| 2026-10-08 | Structured Experiment 4 into detection, early warning and prognosis; prognosis added as a future research question; alert-classification rules defined | Separates distinct problems; recovering episodes must not be counted automatically as false alarms |
| 2026-10-08 | Corrected the expected telemetry behaviour: under congestion throughput rises to capacity and plateaus; under link-quality degradation average latency changes only slightly | The earlier expectation contradicted basic link behaviour. Corrected in the design before any data was generated |
| 2026-10-08 | Synthetic telemetry model specified and generator parameters frozen | Required before detector development |
| 2026-10-08 | Added the Telemetry Integrity & Validation Layer, deterministic defect injection into a copy, and manifest-based evaluation; recorded two simulator simplifications exposed by validation | Telemetry must be validated before detector development; the generator itself is unchanged |
| 2026-10-08 | Added the frozen episode-level train/test split and the Experiment 1 method with expectations recorded before running | Exploratory analysis restricted to training episodes so that it cannot influence decisions about test data |
| 2026-10-08 | Added Experiment 1 observed results, comparison with pre-recorded expectations and interpretation | Results reviewed; mismatches preserved as findings; no parameter, threshold or calculation changed |
| 2026-10-08 | Added the Experiment 2 method, alarm-event and evaluation definitions, and expectations recorded before running | Engineering baseline specified before any detector was run |
| 2026-10-08 | Added Experiment 2 Phase 1 training results and frozen specification (section 12.7) | Recorded before any held-out evaluation; no detector parameter changed |
| 2026-10-08 | Added Experiment 2 Phase 2 held-out results (12.8) and interpretation and limitations (12.9); clarified the Figure 6 selection rule operationally | Single held-out evaluation of the specification frozen at ddd57ca; no detector, threshold or metric changed |
| 2026-10-08 | Added the Experiment 3 method, threshold construction and expectations recorded before running | Interpretable ML detector specified before any model was fitted |
| 2026-10-08 | Added Experiment 3 Phase 1 training results (13.9) and froze the ML specification | Training-only, in-sample results; no model, feature, threshold or rule changed |
