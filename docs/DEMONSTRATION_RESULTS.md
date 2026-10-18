# Verified synthetic demonstration

These measurements were produced by the actual implementation using 2,400 artificial records, seed 42, twenty autoencoder epochs maximum, one hundred controller episodes, window size 64, and Fuzzers held out from development. They are **not UNSW-NB15 measurements and not evidence of real zero-day detection**.

| Method | Precision | Attack recall | F1 | Benign false positive rate |
|---|---:|---:|---:|---:|
| Fixed autoencoder | 0.9023 | 0.9544 | 0.9276 | 0.1692 |
| Random Forest | 0.9877 | 0.9757 | 0.9817 | 0.0199 |
| Supervised MLP | 0.9904 | 0.9362 | 0.9625 | 0.0149 |
| Rule controller | 0.9469 | 0.9210 | 0.9337 | 0.0846 |
| DQN frozen | 0.8991 | 0.9483 | 0.9231 | 0.1741 |
| DQN adapted | 0.9495 | 0.9149 | 0.9319 | 0.0796 |

Random Forest achieves the strongest F1 on this particular artificial distribution. Adaptation lowers the DQN false positive rate but reduces attack recall. The implementation therefore does not demonstrate that DQN is the best model. This is an expected possibility in a research comparison.

The partition sizes were 1,215 training, 304 validation, 351 adaptation, and 530 final-evaluation records. Check `runs/demo/summary.json` for exact configuration and dependency versions, `metrics.csv` for full precision, and `predictions.csv` for individual outcomes. Exported figures are in `runs/demo/figures/`.
