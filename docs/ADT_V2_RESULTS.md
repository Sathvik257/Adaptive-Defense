# ADT v2: completed UNSW-NB15 training and validation

The corrected architecture completed training on 12 November 2026: 30 autoencoder epochs, 20,000 DQN episodes, 256 controller-training flows, seed 42 and Fuzzers withheld from initial training and validation. Runtime was approximately 15.5 minutes on the local CPU. The prepared source contains 151,239 unique flows after conflict removal and deduplication.

An independent checkpoint-reload run evaluated 19,875 development-validation flows and 31,127 test flows without fitting preprocessing, choosing new thresholds or updating model weights. Every test prediction matched the original saved result. Source hashes, saved split order, fingerprint disjointness, Fuzzers exclusion and unchanged checkpoint bytes passed checks. This validates reproducibility of evaluation, not detector effectiveness.

## Development-validation results

This partition was already used for AE checkpoint and static-threshold selection. It is a development result, not an unbiased final estimate. It contains no Fuzzers.

| Model | Precision | Recall | F1 | Accuracy | False-positive rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Static AE | 61.99% | 69.24% | 65.42% | 73.93% | 23.48% |
| ADT with previous-label feedback | 35.78% | 91.71% | 51.48% | 38.43% | 91.04% |
| ADT without feedback | 35.61% | 100.00% | 52.52% | 35.61% | 100.00% |

## Held-out test results

The test contains 12,797 normal and 18,330 attack flows, including 11,250 Fuzzers flows.

| Model | Precision | Recall | F1 | Accuracy | False-positive rate | Fuzzers recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static AE | 74.71% | 47.61% | 58.16% | 59.66% | 23.08% | 33.08% |
| ADT with previous-label feedback | 59.10% | 82.03% | 68.70% | 55.99% | 81.32% | 82.15% |
| ADT without feedback | 58.89% | 100.00% | 74.12% | 58.89% | 100.00% | 100.00% |

Feedback-assisted ADT detected 15,036 attacks and missed 3,294, but incorrectly flagged 10,406 normal flows. Static AE detected 8,727 attacks, missed 9,603 and incorrectly flagged 2,954 normal flows. ADT without feedback classified every flow as an attack. Its apparent 100% recall is therefore degenerate behavior, not successful generalization.

**Conclusion:** the model runs reproducibly, but excessive false positives prevent a practical detection claim. The static baseline has fewer false positives but misses more attacks. A higher F1 alone does not establish superiority. Neither model is validated as a deployment-ready NIDS.

## Protocol limits

- This is custom leave-one-attack-family-out evaluation on UNSW-NB15, not the official train/test split.
- Training and validation use disjoint records. Test records also remain separate, but the same test split was already reported at training completion. This rerun is not a new external test set.
- Feedback-assisted ADT receives the true label after each prediction. Held-out attack labels can therefore affect subsequent states, even with frozen weights.
- Random flow order does not provide the temporal continuity used in the paper's original datasets.
- This is one seed and one held-out family. No multi-seed confidence or deployment claim is supported.
- The full model trained on UNSW-NB15. We have not validated a synthetic-trained model by transferring it to UNSW-NB15.
- Earlier v1 pilot results and the v2 synthetic smoke check are separate experiments.

## Reproduce validation

Use the trusted local run artifacts and original prepared CSV:

```powershell
.venv\Scripts\python.exe -m tools.validate_adt --run runs/adt-v2-unsw-42 --csv data/raw/unsw-consistent.csv --output runs/adt-v2-unsw-42-validation-new
```

The output path must be new. Raw datasets, model checkpoints and per-flow predictions remain local. Published evidence includes [all metrics](adt-v2-results/metrics.csv), [per-family results](adt-v2-results/per_family.csv), [validation checks](adt-v2-results/validation.json) and [training provenance](adt-v2-results/provenance.json).

Next research work should prioritize development-only reduction of false alarms, class-balance and reward analysis, then a predeclared evaluation over multiple seeds and held-out families. Changes chosen after viewing the current test results require new evaluation to avoid overstating generalization.
