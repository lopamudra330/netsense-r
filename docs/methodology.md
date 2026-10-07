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

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-08 | Initial plan | — |
| 2026-10-08 | Renamed `SEVERE_FAILURE` to `SEVERE_DEGRADATION`; defined entry into it as the critical-degradation threshold; reworded the primary question accordingly | "Failure" overstated the condition: the simulated link may still carry traffic |
| 2026-10-08 | Structured Experiment 4 into detection, early warning and prognosis; prognosis added as a future research question; alert-classification rules defined | Separates distinct problems; recovering episodes must not be counted automatically as false alarms |
| 2026-10-08 | Corrected the expected telemetry behaviour: under congestion throughput rises to capacity and plateaus; under link-quality degradation average latency changes only slightly | The earlier expectation contradicted basic link behaviour. Corrected in the design before any data was generated |
| 2026-10-08 | Synthetic telemetry model specified and generator parameters frozen | Required before detector development |
| 2026-10-08 | Added the Telemetry Integrity & Validation Layer, deterministic defect injection into a copy, and manifest-based evaluation; recorded two simulator simplifications exposed by validation | Telemetry must be validated before detector development; the generator itself is unchanged |
