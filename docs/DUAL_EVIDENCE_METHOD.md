# Dual-evidence and feedback-relevance research pilot

## Status and research question

This is a separately implemented and trained research candidate, not a reproduction of ADT and not a verified first-of-its-kind method. Can delayed review feedback be more useful when its influence depends on age and similarity to the current flow's anomaly evidence?

## What is new relative to our ADT implementation

1. Frozen AE reconstruction error plus Ledoit-Wolf Mahalanobis distance in its 100-dimensional linear latent representation. Geometry is fitted on benign training examples only.
2. Each score is mapped to its empirical rank among benign training scores; fixed max fusion combines the ranks. These are not attack probabilities. Independent benign adaptation records calibrate thresholds.
3. A separate DQN with widths 12-32-32-6 chooses one of three thresholds and whether to request the current flow's label. Hard prefix review cap is 5%; requested labels enter the policy state after 32 replay rows.
4. Reviewed confusion evidence is weighted by `exp(-age/128 - EuclideanDistance(current_rank_pair, reviewed_rank_pair)/0.25)`. The last 32 returned reviews are retained. This is similarity in anomaly-score space, not the full latent representation.
5. Matched independently trained ablations remove weighting or replace learned querying with periodic review.

State: two current evidence ranks, their absolute disagreement, two previous-window mean ranks, available query credit (clipped to one), pending-review fraction, age of latest review (clipped), and four weighted confusion fractions. Current truth is absent. Queries have no effect on the already-issued prediction.

Reward during offline training: TP +1, TN +0.1, FP -5, FN -1, query cost -0.02. All known-family training labels are available to compute simulator rewards. Only requested, delayed labels enter the policy state. Test labels are used only as delayed simulated analyst answers and final metrics; no test gradient updates occur.

Threshold actions correspond to benign-calibrated nominal levels 10%, 5%, and 1%. The false-alarm penalty is cost-sensitive, not a hard or proven 5% false-positive constraint. Actual FPR and query usage must accompany each result. It would be misleading to claim equal realized operating budgets merely because a cap exists.

## Experimental controls

Frozen original seed-42 AE and preprocessor. Fuzzers excluded from initial model and controller training. Original partition fingerprints and dataset hash must match. The new geometry and rank reference are fitted on benign training only. The previously unused adaptation normals supply calibration. No adaptation attacks are used.

Pilot training: 300 episodes per variant, 256 sampled known-family training flows per episode, four minibatch updates per episode, batch 128, Adam learning rate 0.001, discount 0.99, target update every 20 episodes, same seed 42 and sampled subsets across variants. No validation-based checkpoint selection; final checkpoint is evaluated. This is 1,200 gradient updates per variant, not evidence of convergence. Existing ADT weights remain unchanged.

The original validation partition was used for AE selection. The test set has already been inspected repeatedly. Results are exploratory, not fresh confirmatory evidence. Row order is shuffled flow replay, not verified chronology or a real drift experiment.

## Prior art and novelty boundary

- [Improving Reconstruction Autoencoder Out-of-distribution Detection with Mahalanobis Distance](https://arxiv.org/abs/1812.02765): reconstruction and latent geometry are established ideas.
- [ConFID](https://www.sciencedirect.com/science/article/pii/S2542660526002088): related intrusion detection with Mahalanobis scoring and conformal calibration.
- [Model update for intrusion detection: delayed labeling and active learning](https://www.sciencedirect.com/science/article/pii/S0167404823003619): label delay and selective acquisition have prior art in NIDS.
- [Optimism and Delays in Episodic Reinforcement Learning](https://proceedings.mlr.press/v206/howson23a.html): delayed feedback is an established RL research problem.

The candidate contribution is the specific coupling of relevance-weighted delayed review state with joint threshold/query control, and an ablation testing that coupling. Neither component names nor combination alone prove novelty. A systematic full-text review and broader experiments are required before claiming a unique method.

## Reproduce

```powershell
.venv/Scripts/python.exe -m tools.evaluate_dual_evidence --run runs/adt-v2-unsw-42 --csv data/raw/unsw-consistent.csv --output runs/my-dual-pilot
.venv/Scripts/python.exe -m tools.evaluate_dual_evidence --run runs/adt-v2-unsw-42 --csv data/raw/unsw-consistent.csv --output runs/my-controller-pilot --controller-episodes 300
```

Use fresh output directories. Local trusted checkpoints and prepared UNSW data are required. Outputs include metrics, per-family detection, score diagnostics, provenance, fitted geometry, and (when requested) new controller checkpoints and training losses. Raw data and checkpoints stay outside version control.

## Required before a publication claim

Multiple training seeds and held-out attack families, independently reserved confirmation data, matched realized FPR/label-cost operating points, no-feedback and age-only/similarity-only ablations, a simple non-RL nearest-reviewed-neighbor comparator, and sensitivity to noisy labels and varying delays. A structured drift stress test must be clearly distinguished from actual chronological traffic. Test results must never be used to select parameters and then called untouched evaluation.


## Measured results (18 November 2026)

31,127 existing test flows; single frozen encoder seed and Fuzzers holdout. Controller pilots each completed 300 episodes/1,200 updates; all 900 recorded losses were finite.

| Method | Precision | Recall | F1 | FPR | Label requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original ADT (full previous-label feedback, prior verified run) | 59.10% | 82.03% | 68.70% | 81.32% | 31,127 |
| Static AE validation-F1 threshold (prior verified run) | 74.71% | 47.61% | 58.16% | 23.08% | 0 |
| Reconstruction, nominal 5% calibration | 72.09% | 9.53% | 16.83% | 5.28% | 0 |
| Dual evidence, nominal 5% calibration | 73.85% | 10.19% | 17.90% | 5.17% | 0 |
| Relevance-weighted learned controller | 68.27% | 1.85% | 3.61% | 1.23% | 1,556 |
| Unweighted learned controller | 68.27% | 1.85% | 3.61% | 1.23% | 511 |
| Relevance-weighted periodic-review controller | 68.27% | 1.85% | 3.61% | 1.23% | 1,556 |

The comparison includes different realized FPRs and information budgets; it is not a leaderboard at matched constraints. The prior ADT/static numbers come from the same verified source/checkpoint/split, not a new retraining run.

Dual evidence modestly improves the reconstruction-only calibrated detector (+0.66 percentage points recall, +1.07 points F1), while absolute detection remains poor. The three controller variants give identical detection metrics despite different review usage. This pilot provides no evidence that relevance weighting helps, and the new controller is not an improved detector. Keep it experimental rather than replacing the original application method. No significance claim or novelty proof is justified.

[Full results](dual-evidence-results/metrics.csv), [family results](dual-evidence-results/per_family.csv), [training summary](dual-evidence-results/training_summary.csv), and [provenance](dual-evidence-results/report.json).

Offline prediction with a trusted method run:

```powershell
.venv/Scripts/python.exe -m tools.predict_dual_evidence --base-run runs/adt-v2-unsw-42 --method-run runs/relevance-controller-nov18 --csv your-flows.csv --output predictions.csv
```

No labels are used by default. Add `--feedback` only to simulate delayed analyst answers from a CSV containing ground truth. The command checks base checkpoint hashes and does not train on input traffic.

The additional [static-threshold controls](dual-evidence-results/static_threshold_controls.csv) show that a fixed nominal 1% dual-evidence threshold has exactly the same confusion counts as all three controllers on both validation and test. The static control uses zero reviews. Therefore this pilot does not justify the controller complexity or review cost. All 71 tests passed, and the prediction command completed a 200-flow no-feedback check.

## Features beyond the paper-based implementation

| Capability | Paper-based ADT implementation in this project | Experimental extension |
| --- | --- | --- |
| Anomaly evidence | Reconstruction score | Reconstruction plus benign latent geometry |
| Controller state | Six rolling-history values | Twelve evidence, budget, pending-review and relevance values |
| Actions | Binary threshold | Three thresholds crossed with query/no-query |
| Feedback | Previous labels in feedback mode | 5% query cap, 32-row delay, age/similarity weighting |
| Analyst output | Score, threshold, decision | Dual scores, decision margin, dominant evidence, reconstruction contributors |
| Controls | Static AE and feedback ablation | Latent-only, dual-only, unweighted and periodic-review variants, matched static thresholds |

These are implementation differences, not proof of scholarly novelty or performance superiority. The original frontend and ADT baseline remain intact.

### Inspectable predictions

The prediction CSV now includes `raw_reconstruction_l2`, `latent_mahalanobis_squared`, the two empirical ranks, `dual_score`, selected `threshold`, `decision_margin`, `dominant_evidence`, and `top_reconstruction_features` (JSON).

Each explanation lists the three transformed feature dimensions with largest squared residual, their residual values, and their fraction of total squared reconstruction error. Numeric contributions are in preprocessed units, and categorical dimensions may be one-hot columns. Fractions are not attack probabilities or causal importance. These explain reconstruction error only, not latent distance or the DQN's action. When latent evidence dominates, the reconstruction contributors must not be presented as explaining the whole decision.

Use `--detector dual-static` to run the calibrated dual-evidence baseline without a controller or review labels. The existing default `--detector relevance` selects the experimental trained controller. This option makes comparisons available without implying that the controller is preferred.

```powershell
.venv/Scripts/python.exe -m tools.predict_dual_evidence --base-run runs/adt-v2-unsw-42 --method-run runs/relevance-controller-nov18 --csv your-flows.csv --output explained-static.csv --detector dual-static
```

Verification on 18 November 2026: all 73 tests passed. Three 200-flow prediction runs covered static, no-feedback controller, and delayed-feedback controller modes. Decision margins matched predictions, top residual shares reconciled with reconstruction error, no-feedback predictions matched the previous command output, and feedback delivery obeyed the 32-row delay and 5% prefix cap. No new accuracy claim follows from these software checks.
