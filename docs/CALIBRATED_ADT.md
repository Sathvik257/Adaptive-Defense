# Research extension: calibrated ADT with budgeted delayed feedback

## Research question

Can a frozen ADT policy retain useful held-out attack recall under an explicit false-alarm operating point when analyst labels are limited and delayed? Does it outperform an identically calibrated autoencoder without RL?

This is an experimental extension, not an established novel algorithm or publication-ready result. Original ADT remains unchanged. The frontend is unchanged.

## What was added

1. A benign-only calibration gate using the order statistic at `ceil((n+1)*(1-alpha))`, with strict score comparison and infinity for insufficient calibration size. Raw reconstruction scores avoid the information loss of clipping to one.
2. A strict prefix query budget: at most `floor(budget * observed_flows)` label requests. A periodic query schedule is label-independent, providing a simple review-policy baseline.
3. Feedback delivered only after a configured number of replay rows. Score statistics use the last k flows; confusion features use the last k delivered reviews of final decisions. This state differs from ADT and is explicitly experimental. The existing policy is frozen, not retrained for the modified state.
4. Matched calibrated-AE ablation and a sweep of 1%, 5%, 10% nominal false-alarm levels; feedback budgets 0%, 5%, 10%, 100%. Original full-feedback ADT and static AE remain references.

An alarm requires BOTH the ADT alarm action and a score above the calibration threshold. Thus it cannot produce more false alarms than the same calibrated AE gate on identical data, but it may lose recall. This algebraic fact is not proof of learned improvement.

## Data discipline

The pilot reuses the existing seed-42 UNSW-NB15 checkpoint and Fuzzers holdout. It calibrates only on benign records from the previously unused adaptation partition. No adaptation attacks are used. Dataset hash and original partition order are checked. Checkpoint hashes must remain unchanged. Test labels are used for metrics and simulated post-decision analyst answers only. This test set has already been examined, so final publication needs fresh predeclared runs.

The conformal order-statistic interpretation assumes exchangeability between calibration and future benign scores. It is marginal, not a promise about every batch or traffic drift. Delays are replay rows, not seconds. Periodic review is simulated and is not a real analyst study.

## Run

```powershell
.venv/Scripts/python.exe -m tools.evaluate_calibrated_adt --run runs/adt-v2-unsw-42 --csv data/raw/unsw-consistent.csv --output runs/calibrated-adt-pilot-42
```

Choose a fresh output directory on repeat runs. Local datasets and checkpoints are required. Aggregate results may be published; flow records must remain excluded.

## Related work and novelty boundary

- [ADT (2023)](https://arxiv.org/abs/2312.01488): reference agent-based threshold control.
- [Online Adaptive Anomaly Thresholding with Confidence Sequences, ICML 2024](https://proceedings.mlr.press/v235/sun24h.html): adaptive thresholding with statistical error control already exists.
- [Operationally calibrated open-world zero-day intrusion detection using hybrid conformal abstention](https://link.springer.com/article/10.1007/s11416-026-00692-3): benign calibration and operating-point constraints already appear in intrusion detection.
- [Dual-Threshold Conformal Deferral for Trustworthy Security Alert Triage](https://www.mdpi.com/2079-9292/15/18/4084): conformal triage and RL investigation have related prior art.

Therefore neither combining conformal calibration and RL nor adding a query budget is claimed as first-of-its-kind. The candidate contribution is an auditable evaluation of ADT under joint false-alarm, feedback-budget and feedback-delay constraints, including its failure modes. A deeper systematic literature review is still needed.

## Before claiming a publishable method

- Train the new state representation explicitly, with known-family training data only; compare against the frozen-policy extension.
- Predeclare a primary operating point and assess every feasible held-out family with multiple training seeds, not just feedback schedules.
- Compare against calibrated AE, original ADT, simple quantile thresholds, and a non-RL budgeted policy.
- Report false-positive rate, held-out recall, precision, F1, review cost, uncertainty across seeds, and paired comparisons. Do not select the best test setting after inspection.
- Assess order sensitivity and distribution shift separately; random flow replay is not streaming evidence.
- Publish negative results when RL adds no value. An informative benchmark is preferable to an unsupported superiority claim.

## Measured pilot (14 November 2026)

Existing frozen seed-42 model, Fuzzers held out; 31,127 test flows. The 5% feedback setting requests 1,556 labels, with 1,554 delivered before replay ends; delay is 32 rows. All settings are retained in [metrics.csv](calibrated-adt-results/metrics.csv), with checkpoint and data provenance in [report.json](calibrated-adt-results/report.json).

| Method | Attack recall | F1 | False-positive rate | Fuzzers recall |
| --- | ---: | ---: | ---: | ---: |
| Original ADT, full previous-label feedback | 82.03% | 68.70% | 81.32% | 82.15% |
| Static AE, validation-F1 threshold | 47.61% | 58.16% | 23.08% | 33.08% |
| Calibrated AE, nominal 5% | 9.53% | 16.83% | 5.28% | 9.37% |
| Calibrated ADT, nominal 5%, 5% reviews, delay 32 | 9.49% | 16.78% | 5.27% | 9.33% |

The gate reduces false alarms at a severe recall cost. The frozen RL policy adds no demonstrated benefit over the matched calibrated AE. Nominal 5% is not a hard sample-level bound: the measured false-positive rate is slightly above it. These results do not support a superiority claim. The next research step is a better anomaly representation and explicitly trained constrained policy, followed by preregistered multi-family/multi-seed evaluation; a gate alone does not solve score overlap.

Validation: the existing 60-test suite passed with the first three extension tests; an additional delayed-history test was then added and checked in the focused suite. The initial default-temp test run encountered a Windows temporary-directory permission error; a project-local temporary directory resolved it.

## Additional features (14 November 2026)

### Boundary-based analyst review

`gated_replay(..., review_strategy="boundary")` selects the score closest to the fixed alarm threshold from flows observed since the previous query. Selection occurs only when the prefix budget permits another query. The feedback delay begins at query time, not at the original flow time. Labels are never consulted to choose a review. The trace records both request time and selected flow index. This is boundary proximity, not a calibrated uncertainty estimate. Periodic review remains the baseline. Selected reviews are biased and their confusion counts must not be treated as unbiased population error estimates.

### Score-distribution warnings

`adaptive_defense.score_monitor.monitor_scores` compares completed, non-overlapping score windows against a fixed reference using empirical CDF distance. Defaults: window 512, warning distance 0.2. These are explicitly heuristic settings, not validated optimal settings or sequential statistical guarantees. The monitor does not request labels, alter thresholds, or update model weights. A warning may mean benign changes, attacks, or a changed class mixture. Comparing shuffled UNSW test traffic to a benign reference does not establish temporal drift detection.

### Per-family reporting

The evaluation now exports `per_family.csv` alongside metrics and provenance. Alert rate means false-positive rate for Normal, and recall for attack families. Only the configured held-out family is unseen during training; results for other families are not zero-day results.

Run the fixed-setting strategy comparison:

```powershell
.venv/Scripts/python.exe -m tools.evaluate_calibrated_adt --run runs/adt-v2-unsw-42 --csv data/raw/unsw-consistent.csv --output runs/calibrated-review-nov14 --focused
```

Outputs include `metrics.csv`, `per_family.csv`, `score_shift.csv`, and `report.json`. Omit `--focused` for the full operating-point sweep. This does not retrain the policy, change the web interface, or establish new-method superiority.


### Review-strategy results

Both strategies requested 1,556 labels (1,554 delivered), with budget 5%, delay 32 rows, and nominal false-alarm level 5%.

| Strategy | Attack recall | F1 | False-positive rate | Fuzzers recall |
| --- | ---: | ---: | ---: | ---: |
| Periodic | 9.49% | 16.78% | 5.27% | 9.33% |
| Boundary proximity | 8.90% | 15.84% | 4.99% | 8.77% |

Boundary review reduces false alarms but loses recall and F1. It remains opt-in; periodic is still the default. This result does not establish superiority. [Aggregate outputs](calibrated-review-results/metrics.csv) include [per-family results](calibrated-review-results/per_family.csv) and [score-shift warnings](calibrated-review-results/score_shift.csv). All 63 tests passed. No frontend changes were made.
