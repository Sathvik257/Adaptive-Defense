# Paper alignment: ADT for Adaptive Defense

**Base paper:** Xue Yang, Enda Howley and Michael Schukat, *ADT: Agent-based Dynamic Thresholding for Anomaly Detection*, ALA 2023, [arXiv:2312.01488v1](https://arxiv.org/abs/2312.01488v1).

> **Architecture audit update:** new v2 runs use the verified later-author topology. See [the full audit](ADT_AUDIT.md) for paper-versus-code discrepancies, corrected defaults and remaining reproduction gaps. The pilot below is a historical v1 result.

## What the project now implements

The new `adaptive_defense.adt_experiment` command implements the paper's central autoencoder + ADT method as an explicitly labelled **UNSW-NB15 flow adaptation**. The original project controller and saved demo remain a separate legacy baseline. Its eight inputs and three actions are not the paper's model.

| Paper component | Implementation |
| --- | --- |
| Section 3.1: normalize inputs to 0–1 | Training-fitted MinMaxScaler for numeric features, one-hot categorical features; clip numeric evaluation features to the fitted range |
| Section 3.1: stride-one windows, any-anomaly window label | `adt.windows`; flow experiment uses tau=1 because the prepared UNSW records have no verified chronological order |
| Equation 4: benign-only AE, L2 reconstruction error | Dense AE trained with mean per-record L2 norm; best benign-validation checkpoint |
| Equations 5–6: mean and sample variance of previous k scores | History contains only completed predictions; default k=2 |
| Section 3.3.1: TP, TN, FP, FN fractions | Counts over the previous k decisions, divided by k; requires true feedback after each prediction |
| Section 3.3.2: actions 0 and 1 | Action is the threshold itself; anomaly iff score > action |
| Equation 8: weighted confusion counts | 0.9 × (TP − FP − FN) + 0.1 × TN over the updated rolling history |
| Algorithm 1 | Replay memory, target network, epsilon-greedy exploration, one MSE gradient update at episode end; target sync every C episodes (default 100) |
| Inference | l=1, frozen model weights; feedback and no-feedback runs reported separately |
| Section 4.2 static baseline | Select threshold maximizing validation F1, never test F1 |

The code stores a transition after **each step**, interpreting Algorithm 1's surrounding prose rather than its apparently misplaced printed loop boundary. Reward is calculated after the action updates the history. Initial history is empty: means/variance are zero initially, available score statistics are used during warm-up, and confusion fractions keep the k denominator. All test records, including startup, count toward metrics.

## What is an adaptation rather than a reproduced detail

The paper evaluates Yahoo A1Benchmark, SWaT and WADI, **not UNSW-NB15**. Its Table 1 uses 20,000 episodes, tau=10/12/10, k=2, l=1/10/10 and alpha/beta=0.9/0.1. This project's flow CLI intentionally fixes tau=1. Do not interpret random adjacent flows as contiguous attack segments.

The PDF does not specify hidden-layer widths. After locating the author's later public code, v2 uses an input-dependent AE with sigmoid output and a 6–32–32–16–2 Q-network. Source revision, layers, activations and paper settings are saved in the architecture manifest. We retain the PDF's L2 scoring, sample variance and weighted reward where the later code differs. See [the audit](ADT_AUDIT.md) for full provenance.

New default runs use 20,000 episodes, k=2, l=1 for unordered flows, 256 controller flows, gamma=.99, DQN learning rate=.0001 and target sync every 100 episodes. The historical v1 pilot used 500 episodes and different hidden layers; its results cannot describe v2. Neither an UNSW experiment nor matching the episode count reproduces the original temporal datasets.

DSPOT, the paper's other baseline, and the three original datasets have **not** been reproduced. Existing Random Forest/MLP results belong to the legacy experiment and must not be mixed with a differently preprocessed ADT result as a controlled comparison.

## Feedback is a research assumption

The paper's TP/TN/FP/FN state cannot be computed from unlabeled live traffic. Paper-mode evaluation here predicts a flow, then reveals its label for the next state. This represents immediate analyst/oracle feedback, not autonomous detection. No current or future label is supplied to the action selector.

Report these results together:

- **Static AE (validation F1):** label-free test inference.
- **ADT (previous-label feedback):** true labels become available after each prediction; weights stay frozen.
- **ADT (no-feedback ablation):** identical trained model, confusion features masked to zero; this is a distribution shift, not a newly trained label-free method.
- **Previous-label baseline:** predict the previous true label, initially normal. This reveals how much label persistence alone explains.

Held-out Fuzzers are absent from initial training and validation. During feedback evaluation, their labels can enter subsequent history. Consequently, held-out recall in this mode is **feedback-assisted** and cannot establish wholly label-free zero-day generalization. The existing separate adaptation partition is unused in this experiment.

## Run on your PC

From the project directory, using the existing Python environment:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_adt.py -q
.venv\Scripts\python.exe -m adaptive_defense.adt_experiment --demo --output runs/adt-demo --epochs 10 --episodes 500
```

For a new installation, first follow the root README's Python setup. No extra Python dependencies are needed for ADT.

Prepare UNSW-NB15 using [the dataset guide](UNSW_NB15.md). Do not substitute uncleaned conflicting records. Then:

```powershell
.venv\Scripts\python.exe -m adaptive_defense.adt_experiment --csv data/raw/unsw-consistent.csv --holdout Fuzzers --output runs/adt-unsw-fuzzers-42 --seed 42 --epochs 10 --episodes 500 --controller-rows 256
Start-Process runs/adt-unsw-fuzzers-42/report.html
```

Use a fresh output directory for every run. `--episodes 20000` matches the paper's episode count only; it does not turn the flow adaptation into an exact reproduction. Repeat with several seeds and held-out families before drawing research conclusions.

## Outputs and compatibility

Every completed run contains a readable `report.html`, metrics, per-flow predictions, feedback/no-feedback state traces, AE and controller training histories, a partition manifest, controller subset manifest, train-fitted preprocessing and model checkpoints. `adt_summary.json` records configuration, hashes, timing, normalization and limitations. It is intentionally distinct from legacy `summary.json`: the old dashboard and inference loader must not silently load incompatible six-state/two-action weights.

The saved pickle is a trusted local artifact, not a safe upload format. Raw datasets and trained weights remain local. The existing web console continues to show its original recorded experiment; it does not claim to serve the new ADT model.

## Brief explanation for a presentation

“Our base paper is ADT by Yang, Howley and Schukat. We train an autoencoder on normal traffic and use its reconstruction error as an anomaly score. A DQN then chooses an active or passive threshold using recent scores and feedback about earlier decisions. We adapted the method to UNSW-NB15 and evaluate it against a static threshold, with and without feedback. We keep unfamiliar attack families out of initial training, but report feedback-assisted results separately because the paper's state needs previous labels.”

## Historical v1 UNSW-NB15 pilot (seed 42)

The first run used 10 AE epochs, 500 ADT episodes, 256 controller-training flows and held out Fuzzers. Evaluation used 31,127 test flows. This is an early pilot, not a converged reproduction.

| Model | F1 | Held-out Fuzzers recall | False-positive rate |
| --- | ---: | ---: | ---: |
| Static AE, validation-F1 threshold | 0.4960 | 0.1548 | 0.0625 |
| ADT, previous-label feedback | 0.0000 | 0.0000 | 0.0000 |
| ADT, no-feedback ablation | 0.0000 | 0.0000 | 0.0000 |
| Previous-label baseline | 0.5900 | 0.5892 | 0.5872 |

**The trained ADT chose the passive threshold for every test flow and missed all attacks.** This is a failed detection result, not a successful zero-day detector. The previous-label baseline's high false-positive rate also makes its higher F1 insufficient evidence of usefulness. Random flow order lacks the contiguous segments on which the paper relies, and this short run does not resolve convergence. Do not tune against these test results or present the paper's published scores as ours.

See [saved metrics](adt-pilot/metrics.csv), [provenance](adt-pilot/provenance.json) and the [readable report](adt-pilot/report.html). A next research experiment should predefine training settings, use development-only tuning and repeat across seeds; exact paper replication additionally requires the original temporal datasets and DSPOT.

## Predict from the saved model

For a CSV with the same feature schema, without labels:

```powershell
.venv\Scripts\python.exe -m adaptive_defense.adt_predict --run runs/adt-unsw-fuzzers-42 --csv your-flows.csv --output predictions/adt.csv
```

This is explicitly the no-feedback ablation. For offline evaluation only, append `--feedback` and supply a binary `label` column to reveal each label after its prediction. Each invocation resets the history; it is not a continuous live service. Only load checkpoints you created or trust.
