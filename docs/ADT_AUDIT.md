# ADT paper audit and architecture correction

## Sources and reproducibility boundary

The controlling source is the user-supplied **2312.01488v1.pdf**, *ADT: Agent-based Dynamic Thresholding for Anomaly Detection* (Yang, Howley and Schukat, 2023). [Paper](https://arxiv.org/abs/2312.01488v1).

The architecture was additionally checked against [the author's later public repository](https://github.com/XueYang0130/ADT/tree/4b291b1b9c05c748f61882921a1aab86101f0954), revision `4b291b1b9c05c748f61882921a1aab86101f0954`. Its README refers to two 2024 journal papers. It is **not established as the exact code used for the 2023 paper**. Architecture facts come from `adt/models/ae.py` and `adt/models/dqn_agent.py`; training settings come from `adt/training/dqn_training.py`.

The PDF does not disclose hidden-layer widths or activations. Our earlier fixed AE and 6–64–64–2 controller were implementation choices, not verified author architectures. New runs now use an independently written PyTorch implementation of the later author's topology:

```text
Window of tau readings, m features → flatten to D = tau × m
AE: D → floor(D/2) → floor(D/4) → Z → floor(D/4) → floor(D/2) → D
     ReLU           ReLU         linear  ReLU          ReLU       sigmoid
L2 reconstruction score → frozen training-based normalization → [0,1]
Previous k scores + previous prediction outcomes → six-state vector
Q network: 6 → 32 ReLU → 32 ReLU → 16 ReLU → 2 linear Q values
argmax → threshold {0,1} → compare current score → prediction → feedback
```

Author code uses Z=100 for multivariate datasets and Z=3 for Yahoo. The UNSW flow adaptation defaults to Z=100 with an explicit override. Since tau=1 there, the input dimensions differ from the paper datasets. Equal topology does not imply equal learned weights or performance. PyTorch initialization and optimizer numerical behavior are not claimed to be identical to TensorFlow/Keras.

## Section-by-section verification

| Source | Verified requirement | Project status |
| --- | --- | --- |
| Abstract, Sections 1–2 | AE representation/scoring + DQN threshold control, not an attack-family classifier | Matched conceptual architecture; no claim of reproducing cited related-work models |
| Equation 2 | Terminal target r; otherwise r + gamma max target-Q | Implemented and terminal bootstrap tested |
| Equation 3 | Precision, recall, F1 | Standard metrics plus separately saved paper-convention metrics |
| Section 3.1 | Normalize 0–1; stride 1; n−tau+1 windows; any anomaly labels a window | Window utility tested; UNSW CLI tau=1, not a temporal reproduction |
| Equation 4 | L2 reconstruction norm; benign-only AE | Retained L2 objective and scoring despite later code using MSE |
| Equations 5–6 | Previous k scores, mean and sample variance with denominator k−1; k=1 variance 0 | Matched once history is full; explicit startup policy below |
| Section 3.3.1 | Six values ordered mean, variance, TP/k, TN/k, FP/k, FN/k | Matched PDF order; no current-label access |
| Section 3.3.2 | Thresholds 0/1; anomaly when score exceeds threshold | Strict >, including boundary tests |
| Equation 8 | alpha(TP−FP−FN)+beta TN, alpha+beta=1 | Rolling counts, default .9/.1; reward assigned after current outcome |
| Section 3.3.4 | Hold training action l steps; inference l=1 | Implemented; non-1 inference interval rejected |
| Algorithm 1 | Replay, initially copied target, episode-end update, epsilon decay, periodic target copy | Implemented; one gradient update per episode, copied initialization |
| Section 3.4 | No downsampling before window construction | Utility retains every stride-one window; flow training subset is a separate ADT-training choice |
| Section 4.1 | Yahoo, SWaT, WADI; small mixed-class ADT subsets | Not reproduced on those datasets; actual UNSW subset fraction saved |
| Section 4.2 | Static and DSPOT baselines | Static validation-F1 baseline implemented; DSPOT remains missing |
| Table 1 | 20,000 episodes; k=2; alpha=.9, beta=.1; tau/l per dataset | Table recorded in code; default episodes corrected to 20,000; UNSW tau=1/l=1 is explicitly an adaptation |
| Section 5.1 | Ten evenly divided subsets; special all-normal convention; two-sided Wilcoxon at .05 | Added reusable ten-block report with sample SD and Wilcoxon comparison to static |
| Section 5.2 | l sweep {1,10,20,50,100}, k sweep {2,10,15,20,50}, reward-weight study | Controls exist in core; published sweeps have not been rerun |
| Section 5.3, Figures 4–5 | Training stability, threshold trace, real SWaT segments | Histories and traces saved; SWaT figure and results not reproduced |
| Section 6 | Isolated-point-anomaly limitation; continuous thresholds future work | Documented; binary actions retained |

Table 1 settings are Yahoo (tau=10,l=1), SWaT (tau=12,l=10), WADI (tau=10,l=10), all 20,000 episodes/k=2/.9/.1. The paper's ADT training fractions are approximately .35%, .11% and .58%, respectively. We do not invent equivalent chronological slices in shuffled UNSW records.

## Resolving paper-versus-code disagreements

The later repository uses MSE scores/loss, population variance, TP/FP/TN/FN ordering, >= comparison and single-step +1/−1 rewards. Those differ from the supplied PDF. We **retain the PDF's L2, sample variance, TP/TN/FP/FN ordering, strict > and weighted rolling reward**. Thus this is “2023 paper equations with later author topology,” not a verbatim copy of either experimental implementation.

The reference training loop assigns `state = next_state` before recording the transition. Our replay stores the actual pre-action state and post-action next state, as required by the paper's Bellman transition. Its data preparation also fits scaling on concatenated normal and attack data; we retain initial-training-only preprocessing to avoid test-distribution leakage. Its sliding-window helper omits the last valid window; ours returns n−tau+1 as specified by the paper. These are deliberate choices, not unnoticed mismatches.

New defaults sourced from the later code: gamma=.99, DQN Adam learning rate=.0001, replay capacity=20,000, target interval=100 episodes, DQN batch=256, AE Adam=.001, AE batch=8,000, AE epochs=30, epsilon linear decay to .001 over .99 of planned episodes. We use the PDF's initially synchronized target and periodic copies after completed episode multiples, not the reference's zero-based first-episode sync. We preserve benign-validation checkpoint selection. Exact equivalence to the author's training procedure is not claimed.

## Explicit unresolved choices

- The paper does not explain how unbounded L2 scores become [0,1]. Our fixed initial-training maximum and clipping remain an adaptation; clipping counts are saved.
- The state equations assume t>k. We initialize empty history and use available score statistics during startup; confusion fractions retain denominator k. Initial predictions remain in the evaluation.
- The timing of the rolling reward is underspecified. We use the updated history after each action, so current feedback affects reward and the next state, never the current action.
- Immediate previous-label feedback is an oracle/analyst assumption, not a deployable label-free capability. The no-feedback ablation is reported separately.
- Wilcoxon comparisons concern correlated replay blocks and are descriptive; p<.05 does not prove superiority. There is no multiple-comparison correction. DSPOT comparisons cannot be claimed until implemented.
- Original datasets, exact training slices, all parameter sweeps and published results remain unreproduced. Exact 2023 hidden architectures cannot be verified from the available sources.

## Artifact versioning

`adt-flow-v2` stores the verified-topology architecture manifest and a 6–32–32–16–2 controller. Prediction still loads `adt-flow-v1` using its original fixed AE and 6–64–64–2 controller. The saved 500-episode UNSW pilot in `docs/adt-pilot` is a **v1 historical pilot** and must not be cited as a result of v2. New training output includes `paper_metrics.csv`, `robustness_blocks.csv` and `robustness_comparison.csv` in addition to the existing artifacts.

Example fresh v2 experiment (20,000 is the new default; may take substantial CPU time):

```powershell
.venv\Scripts\python.exe -m adaptive_defense.adt_experiment --csv data/raw/unsw-consistent.csv --output runs/adt-v2-unsw-42 --holdout Fuzzers --seed 42
```

To inspect the architecture without training:

```powershell
.venv\Scripts\python.exe -c "from adaptive_defense.adt_architecture import architecture_manifest; import json; print(json.dumps(architecture_manifest(612),indent=2))"
```

Here 612 is an illustrative SWaT flattened dimension (12×51), not the UNSW feature dimension. Actual run manifests record the fitted dimension.

## Validation of this correction

The full project suite passed **57 tests**, including layer dimensions, sigmoid output bounds, terminal Bellman targets, feedback causality and paper metric edge cases. An end-to-end synthetic check used two AE epochs and three controller episodes to verify artifact generation and v2 checkpoint reload. It is a software check, not an accuracy benchmark. Reloading the historical v1 checkpoint produced exactly the same 32 smoke-sample predictions as before the change. No new full UNSW v2 training result is claimed.
