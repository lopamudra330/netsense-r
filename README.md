# NetSense-R

### When does a communication network begin to fail — and can we see it coming?

**ML-Assisted Degradation Detection and Reliability Analysis for Communication Networks**

*NetSense-R: the "R" stands for **Reliability**. This is a Python project; it is not related to the R programming language.*

> **Status: in development.** The research questions and experimental plan are fixed
> (see [`docs/methodology.md`](docs/methodology.md)). No experiments have been run yet,
> so this README deliberately contains no results.

## Research statement

Much of my engineering work in communications infrastructure has been about answering
*"Why did this communication fail?"* after the event. NetSense-R asks the next question:
*can observable telemetry show that a communication link is degrading before it fails
severely?*

The study uses a controlled, **synthetic** model of a generic IP access/backhaul link that
degrades through two distinct mechanisms, congestion and link-quality degradation. It
compares conventional threshold-based monitoring with simple, interpretable
machine-learning models (Logistic Regression and Random Forest). Before any prediction,
a validation layer checks whether the telemetry itself can be trusted.

**Primary research question**

> In a controlled, synthetic model of gradual communication-link degradation, how early
> and how reliably can (a) conventional threshold monitoring and (b) simple interpretable
> machine-learning models detect the onset of degradation before the link reaches severe
> failure, and at what cost in false alarms?

Supporting research questions and the experimental design are in
[`docs/methodology.md`](docs/methodology.md).

## Data statement

All telemetry in this repository is synthetic. *This dataset is synthetic and designed
for methodological exploration; it does not represent measurements from a specific
production network.*

## Licence

MIT. See [`LICENSE`](LICENSE).
