# Model card

Intended use: offline educational comparisons of unfamiliar-family detection.
Not intended for autonomous blocking, production security claims or deployment
without representative validation. The console uses a frozen synthetic demo.

The benign-trained autoencoder has a 16-dimensional bottleneck. A DQN changes
thresholds over windows; Random Forest, MLP, fixed AE and rule baselines share
the partitions. Hyperparameters and package versions accompany each run.
PPO is a separate experimental budget simulator, not part of LOAO comparisons.

Evaluate precision, recall, benign false positive rate, held-out recall, known
recall and confusion counts. Summaries show per-family seed means and sample
standard deviations; they are not confidence intervals or independent datasets.
No superiority is assumed. Latency must state hardware, batch size, precision,
runtime and whether preprocessing/serialization are included.

Detectability uses distance to the nearest known attack centroid in the frozen
AE latent space, standardized by training-known-attack coordinate deviations.
It is an explanatory association, not causal evidence. Each fold trains its own
encoder; standardization does not guarantee fold-comparable geometry. Report
per-seed/family points, training convergence and alternative distance definitions.
There is no universal direction of the distance/recall relationship. Both far
and nearby attacks can be detected or missed by different decision rules.
