# Demonstration and viva guide

## Five-minute presentation

1. Explain the problem: unfamiliar traffic may evade known signatures, while an overly sensitive detector produces false alarms.
2. Open Overview and identify whether the selected run is synthetic or a real CSV experiment.
3. Explain the autoencoder: normal records train a reconstruction model; reconstruction error becomes an anomaly score.
4. Explain the DQN: it adjusts the threshold using past-window observations, with rewards from delayed development feedback.
5. Open Model comparison. Show recall and false alarms together. Do not hide stronger baseline results.
6. Open Traffic replay. Explain that the trace is saved offline replay, not a live attack feed.
7. Open Score a CSV. Upload a compatible CSV and show that labels are not required for inference. Download the predictions.
8. Explain the adaptation boundary: separate feedback records, final evaluation untouched, and post-adaptation recall clearly distinguished from unseen recall.

## Common questions

**Why deep reinforcement learning?** It learns a discrete threshold-control policy over windows. Its usefulness must be demonstrated against a fixed threshold and a simple rule.

**Where are the labels?** They supply development rewards and final evaluation truth. They are excluded from features. Final labels never update the tested policy.

**Does it detect all zero-day attacks?** No. Category withholding measures unfamiliar-category generalization, not vulnerability discovery.

**What is adaptive?** The threshold changes over windows, and a cloned policy can be retrained with separate feedback data. These are different forms of adaptation and are reported separately.

**Why not just accuracy?** Many benign records can hide missed attacks. Recall, precision, false alarms and per-category metrics expose the trade-off.

**What if Random Forest wins?** Report it. The project is an experiment, not a guarantee that a more complex method performs better.

**Is it real time?** No. The present project is an offline flow-based prototype. Live deployment requires a compatible feature extractor and additional validation.

**What have you actually measured?** Show metrics.csv and summary.json for the selected run. Synthetic demo numbers are software verification only.
