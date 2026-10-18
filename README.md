# Adaptive Defense

## Reusable research benchmark

Install with `python -m pip install .`, then run `adaptive-benchmark --demo --output
runs/loao`. The default uses five seeds and leaves each attack family out in turn.
See [benchmark guide](docs/BENCHMARK.md), [dataset card](docs/DATASET_CARD.md),
[model card](docs/MODEL_CARD.md), and [isolated lab](lab/README.md).

## React web console

The new React + FastAPI console provides a recorded WebSocket replay, traffic
projection, decision inspector, SQLite analyst queue and benchmark cost controls.

```powershell
python -m pip install -r requirements-api.txt
npm.cmd ci --prefix web
npm.cmd run build --prefix web
python serve_api.py
```

Open http://127.0.0.1:8000. See [console architecture and research boundaries](docs/WEB_CONSOLE.md).
The Streamlit interface described below remains available as a legacy dashboard.

**Zero-Day Attack Detection in NIDS With Deep Reinforcement Learning**

A runnable research project combining a benign-trained deep autoencoder with a DQN threshold controller, supervised baselines, a rule controller, disjoint adaptation experiments, and a Streamlit dashboard.

The workspace has five focused views: **Overview**, **Compare**, **Replay**, **Analyze CSV**, and **About**. Compare models against the fixed-threshold baseline, explore missed attacks and false alarms, and replay saved decisions. The UI and model code are separated into modules; see [the architecture guide](docs/ARCHITECTURE.md).

**Stack:** Python · PyTorch · scikit-learn · pandas · Streamlit · Docker · GitHub Actions.

> The bundled demonstration is synthetic. It verifies implementation, not cybersecurity effectiveness. Real UNSW-NB15 experiments require the dataset CSV files. Withheld categories simulate unfamiliar attacks; they do not establish real-world zero-day detection.

## Quick start on this computer

The local `.venv` has the additional dependencies and uses preinstalled PyTorch, pandas, and Streamlit.

```powershell
.\.venv\Scripts\python -m streamlit run dashboard.py --server.address 127.0.0.1
```

Or run `start-dashboard.ps1`. The dashboard reads completed folders under `runs/`.

To create another demonstration:

```powershell
.\.venv\Scripts\python -m adaptive_defense.experiment --demo --output runs/demo-new --epochs 20 --episodes 100 --window 64
```

## Fresh installation

Use Python 3.11 or newer with a supported PyTorch build. A CPU is sufficient for the small demo.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m adaptive_defense.experiment --demo --epochs 20 --episodes 100 --window 64
.\.venv\Scripts\python -m streamlit run dashboard.py --server.address 127.0.0.1
```

If Windows `ensurepip` fails, install into the created environment with `python -m pip --python .venv\Scripts\python.exe install -r requirements.txt`. Do not copy another machine's `.venv`.

## Real UNSW-NB15 experiment

Get the two prepared CSVs using [the official dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). See `data/raw/README.md`.

```powershell
.\.venv\Scripts\python -m adaptive_defense.experiment --csv data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv --holdout Fuzzers --output runs/unsw-fuzzers-42 --seed 42 --epochs 40 --episodes 100 --window 256
```

Repeat with seeds 7 and 123 and separate output folders. Repeat with another held-out category in a separate experiment. Do not select hyperparameters using those final results. The program refuses to overwrite a completed run.

## What is implemented

| Component | Implementation |
|---|---|
| Data integrity | Binary-label validation, feature deduplication, conflicting-label rejection, saved split manifest |
| Preprocessing | Training-only imputation, scaling, categorical encoding with unknown-category support |
| Autoencoder | Dense 128–64–16–64–128 network, benign-only optimization, early stopping |
| DQN | 8-input state, three actions, replay buffer, target network, Huber loss, gradient clipping |
| Baselines | Random Forest, supervised MLP, fixed autoencoder, simple alert-rate controller |
| Evaluation | Frozen known/unseen recall, F1, precision, false positives, confusion counts |
| Adaptation | Separate feedback partition; adapted policy evaluated without test-label feedback |
| Dashboard | Metrics, category comparisons, learning curves, threshold replay, CSV inference/download |

## Data boundaries

Known categories and normal traffic: 60% training, 15% validation, 10% adaptation, 15% final evaluation (rounding applies). Held-out categories: 40% adaptation and 60% evaluation, zero training/validation. Identical feature vectors are deduplicated before partitioning. These are custom partitions, not the official benchmark split. Session-level isolation cannot be established from these simplified CSV records alone. Replay ordering is synthetic, not chronological.

The autoencoder is trained only on benign training rows. The preprocessor is fitted on training features. The controller learns rewards on development rows. Validation chooses its checkpoint. Final labels are used only by the evaluator. Adaptation uses a cloned policy and separate records; the autoencoder remains frozen.

## Outputs

Each completed run includes:

- `summary.json`: configuration, versions, provenance, audit and limitations.
- `metrics.csv`: comparison metrics for six model variants.
- `predictions.csv`: test labels, categories, scores and predictions.
- `split_manifest.csv`: reproducible record fingerprints and assigned partitions.
- `autoencoder_history.csv`, `controller_history.csv`: training histories.
- `threshold_trace.csv`, `adapted_threshold_trace.csv`: window decisions.
- `pipeline.pkl`, `autoencoder.pt`, `dqn.pt`, `dqn_adapted.pt`: inference artifacts.

Only load local artifacts you trust: Python pickle is not safe for untrusted files. Uploaded CSV data is processed locally and is not sent to a service.

## Verification

```powershell
.\.venv\Scripts\python -m pytest -q
```

Tests check target exclusion, split isolation, duplicate handling, unknown categorical features, hidden evaluation labels, delayed feedback, reward calculations, terminal boundaries, and dashboard rendering. A full demonstration run exercises training and artifact generation.

In a restricted Windows environment, set `TMP` and `TEMP` to a writable workspace directory and pass a fresh `--basetemp` under that directory if the system temporary directory is blocked.

## Command-line inference and exportable figures

```powershell
.\.venv\Scripts\python -m adaptive_defense.predict --run runs/demo --csv runs/demo/synthetic_input.csv --mode frozen --output runs/demo/new_predictions.csv
.\.venv\Scripts\python -m adaptive_defense.figures runs/demo
```

Inference accepts feature-only CSVs. Available modes are `fixed`, `frozen`, and `adapted`. Figure export produces PNG model-comparison, threshold-replay, and learning-curve images with an explicit synthetic-data label when applicable.

## Read next

- `docs/PROJECT_REPORT.md`: detailed academic report and methodology.
- `docs/DEMO_GUIDE.md`: demonstration script and viva questions.
- `docs/HOSTING.md`: Docker and Python hosting, resource limits and deployment instructions.
- `docs/resource-check.json`: measured local resource and inference checks.
- `docs/ARCHITECTURE.md`: module map, stack choices and development workflow.
- `docs/DAILY_SYNC.md`: daily synchronization behavior and availability requirements.

## Host the dashboard

```sh
docker compose up --build -d
```

Open `http://localhost:8501`. For a Python hosting provider, install `requirements-host.txt` and use `python serve.py` as the start command. See the hosting guide before exposing the service publicly. No public deployment is included.

## Scope

This is an offline NIDS research prototype. It does not capture live traffic, block traffic, identify a vulnerability/CVE, or guarantee discovery of zero-day exploits. A stationary synthetic replay may favor a fixed threshold; DQN superiority must be measured rather than assumed. No automatic alert budget or live analyst interface is claimed.
