# Related work and improvement review

Reviewed 2 November 2026. These are related systems, not evidence that this project outperforms them.

| Source | What it contributes | Decision for Adaptive Defense |
| --- | --- | --- |
| [Kitsune / KitNET](https://github.com/ymirsky/Kitsune-py) and [paper](https://arxiv.org/abs/1802.09089) | Online network anomaly detection using an ensemble of autoencoders and incremental traffic features | Keep the autoencoder baseline, distinguish recorded flow replay from live packet extraction. A comparable KitNET baseline needs the same splits and compatible feature pipeline; no invented comparison scores. |
| [River ADWIN](https://riverml.xyz/dev/api/drift/ADWIN/) | Adaptive-window monitoring for stream distribution changes | Add separate descriptive score monitoring, without treating a shift as an attack. Our implementation is a fixed-reference two-sample empirical KS distance, NOT ADWIN. |

## Implemented improvement

GET /api/monitoring?count=128&window=64 compares the first 64 observed scores with the last 64 observed scores. Until two disjoint windows exist it reports warming-up. Distance is the maximum absolute difference between the empirical cumulative distributions, ranging from zero to one. Medians provide interpretable context. Only the supplied replay prefix is used; future observations and ground-truth labels are excluded. Invalid bounds and nonfinite observed scores are rejected.

This is a diagnostic, not a calibrated drift detector. The initial window is not guaranteed benign. Correlation, attack prevalence, changing workloads and model revisions can all affect the distance. There are no p-values, automatic response rules, retraining or attack claims. Reset replay to establish a fresh comparison. Training and held-out results are unchanged.

## Highest-value work still needing evidence

1. Acquire a licensed real dataset and rerun identical LOAO splits across AE, DQN, RF and MLP; report false alarms and cost alongside recall.
2. Collect isolated lab traffic and document feature compatibility before claiming external validation.
3. Evaluate a genuine streaming detector with controlled shifts, detection delay and false-alarm rates before automatic adaptation.
4. Add KitNET only with matched data and fair preprocessing; importing another architecture alone is not a research result.

## Frontend decision (2 November 2026)

The user chose to retain the original frontend. The alternative design remains only in the local codex/frontend-review worktree for reference; do not push or merge it. Daily synchronization applies to the main checkout only. The backend monitoring endpoint and research improvements are retained independently of the rejected design. Existing production frontend files remain unchanged.
