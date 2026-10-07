# Methodology

This document records the research questions and experimental plan for NetSense-R.
It was written **before** any data was generated or any model was trained. Later
changes to the plan will be recorded in the [change log](#change-log) at the end,
with the reason for each change.

## 1. Motivation

Operational monitoring of communication systems is often reactive: an alarm fires once
a service has already failed, and engineers then work backwards to find out why.
NetSense-R asks whether the telemetry that a link already produces (delay, loss,
retransmissions, throughput) shows a recognisable *signature* of developing degradation
early enough to act on it.

The engineering motivation comes from my professional experience investigating
communication failures in the UK smart-metering domain. **The study itself does not use,
model or represent any smart-metering, DCC, Arqiva or other proprietary or production
system.** It models a generic IP access/backhaul link.

## 2. Research questions

### Primary question

> In a controlled, synthetic model of gradual communication-link degradation, how early
> and how reliably can (a) conventional threshold monitoring and (b) simple interpretable
> machine-learning models detect the onset of degradation before the link reaches severe
> failure, and at what cost in false alarms?

The question is about method. Because the data is synthetic, the answers describe how
the detectors behave **under the stated assumptions of the simulation**. They do not
claim how detectable degradation is in real networks.

### Supporting questions

| | Question | Addressed by |
|---|---|---|
| **RQ1** | Which telemetry variables change earliest and most clearly as a link moves from normal operation into early degradation, and does this depend on the *cause* of degradation? | Experiments 1 and 4 |
| **RQ2** | When is simple threshold logic sufficient, and when (if ever) do Logistic Regression or Random Forest add detection value, especially for early degradation? | Experiments 2 and 3 |
| **RQ3** | What kinds of errors remain (false alarms, missed early degradation, confusion between neighbouring states), how well do detectors generalise to a degradation cause they were not trained on, and why are these cases difficult? | Experiments 3 and 4, robustness check, failure-case analysis |

## 3. Terminology

- **Degradation detection**, not anomaly detection. The ML experiments are
  *supervised*: models learn from examples labelled with the network state.
  Unsupervised anomaly detection, which learns without labels and could in principle
  flag unfamiliar failure types, is identified as future work.
- **Network states**: `NORMAL`, `EARLY_DEGRADATION`, `DEGRADED`, `SEVERE_FAILURE`.
- **Degradation causes**:
  - *Congestion*: offered traffic approaches or exceeds link capacity, so queues
    build up in network buffers.
  - *Link-quality degradation*: the transmission link itself deteriorates (for
    example interference or a failing interface), so frames are corrupted.

## 4. Study design overview

The details of the telemetry model are documented in Step 2, together with the
justification for every parameter.

1. **Synthetic telemetry.** Independent *episodes* of a single link, sampled once per
   minute. Each episode has a hidden degradation severity that changes over time. State
   labels are derived from that hidden severity, **never** from the observed metrics.
   This prevents the labels from simply encoding a threshold rule.
2. **Episode types.** Stable operation that includes harmless traffic bursts;
   degradation that progresses to severe failure; and degradation that recovers.
   Recovering episodes make false alarms measurable.
3. **Telemetry Trust Layer.** Telemetry is validated (missing values, duplicate
   timestamps, impossible values, gaps, inconsistent states) before any detector
   sees it.
4. **Train/test separation by episode.** Whole episodes are assigned to either
   training or testing, so neighbouring minutes of the same episode never appear in
   both.

## 5. Experiments

| Exp | Question | Method | Metrics |
|---|---|---|---|
| 1. Behaviour | What does degradation look like in telemetry, and does it differ by cause? | Per-state distributions, averages and standard deviations; correlation between variables | Descriptive statistics; overlap between neighbouring states |
| 2. Engineering baseline | How well does conventional threshold monitoring perform? | Documented rules on rolling averages; thresholds derived from the training data's normal periods | Per-class precision, recall, F1; confusion matrix |
| 3. ML comparison | Do Logistic Regression and Random Forest add value over thresholds? | Raw metrics plus simple rolling features; class weighting; repeated over several random seeds | Per-class and macro F1, recall, confusion matrix, mean ± standard deviation across seeds |
| 4. Early warning | Can degradation be detected before severe failure, and which signals give the earliest warning? | Episode-level evaluation of first-alert time against failure-onset time | Lead time (minutes), share of failures warned in advance, false alarms per 24 h of normal operation |
| Robustness | Do detectors generalise to an unseen degradation cause? | Train on congestion episodes only; evaluate on link-quality episodes | As Experiments 3 and 4 |
| Failure cases | Where and why do detectors fail? | Inspection of false positives, false negatives and confused states | Error counts by state and cause; annotated examples |

**Why accuracy is not the primary metric.** Most minutes of a realistic link are
normal. A detector that always answers "normal" would therefore score high accuracy
while detecting nothing. Precision, recall and F1 per class show what accuracy hides.

## 6. Research integrity commitments

1. Synthetic data is never presented as real measurement.
2. Generator parameters are fixed before experiments are run. They are not adjusted
   after seeing model performance in order to obtain better-looking results. If a change
   becomes scientifically necessary (for example, a modelling error), the reason is
   explained before the change is made, and the change is recorded in the change log
   below.
3. Negative or weak results, including poor generalisation in the robustness check,
   are reported as results.
4. Results are reported with variation across random seeds and without false
   precision.

## Change log

| Date | Change | Reason |
|---|---|---|
| — | Initial plan | — |
