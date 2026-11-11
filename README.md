# Adaptive Defense

### Zero-Day Attack Detection in NIDS With Deep Reinforcement Learning

An offline network intrusion detection research project that combines a benign-trained autoencoder with a Deep Q-Network (DQN) threshold controller. Explore recorded model decisions, compare baselines, and evaluate detection of attack families withheld from initial training.

**[Open the live demo](https://web-flame-eight-75.vercel.app/)** · [Research results](docs/RESEARCH_RESULTS.md) · [Benchmark guide](docs/BENCHMARK.md) · [Deployment guide](docs/VERCEL.md)

> **Research status:** the hosted demo uses synthetic traffic. A separate UNSW-NB15 pilot and a paper-aligned ADT experiment are available locally. Held-out families are a proxy for unfamiliar attacks, not proof of real-world zero-day detection. The project does not capture or block live traffic.

## Base paper and new ADT experiment

The project now includes an implementation of **[ADT: Agent-based Dynamic Thresholding for Anomaly Detection](https://arxiv.org/abs/2312.01488v1)** (Yang, Howley and Schukat, 2023): benign-only L2 autoencoder scoring, a six-feature history state, binary threshold actions and episode-end DQN updates.

Read the **[architecture audit](docs/ADT_AUDIT.md)** for verified author layer sizes and differences between the supplied paper and later source code. New ADT runs use the corrected topology; the previous pilot remains labelled v1.

Read the **[paper alignment and run guide](docs/ADT_PAPER.md)** for exact equations, UNSW-NB15 adaptations and feedback assumptions. The new experiment reports static-threshold, feedback-assisted ADT, no-feedback ADT and previous-label baseline results. Its models are separate from the original dashboard's three-action controller.

```powershell
.venv\Scripts\python.exe -m adaptive_defense.adt_experiment --demo --output runs/adt-demo --epochs 10 --episodes 500
```

## Explore the demo

The [hosted console](https://web-flame-eight-75.vercel.app/) uses the original dark interface and requires no installation.

1. Start the recorded replay and adjust its speed.
2. Select a flow to inspect its anomaly score, threshold action and reconstruction errors.
3. Request analyst review and assign a label within the 20-query demonstration budget.
4. Open **Benchmarks** to compare saved results, detection trade-offs and measured research evidence.

The Vercel version runs entirely in the browser. Review labels stay in that tab's session storage; they are not sent to a server or used to retrain a model. The hosted site performs no live model inference or training.

## How the original demo works

```text
Traffic records → Train-fitted preprocessing → Autoencoder reconstruction score
                                                    ↓
                                          DQN threshold controller
                                                    ↓
                                         Allow / alert decision
                                                    ↓
                                      Evaluation and analyst review
```

The autoencoder learns benign patterns. Reconstruction error provides an anomaly score. The DQN chooses whether to lower, hold or raise the decision threshold, balancing missed attacks and false alarms in a simulated environment. Supervised and non-RL baselines test whether the controller actually helps.

| Component | Implementation |
| --- | --- |
| Representation | Dense autoencoder with a 16-dimensional latent space; benign-only training |
| Controller | DQN with an eight-value state, three actions, replay buffer and target network |
| Baselines | Random Forest, supervised MLP, fixed-threshold autoencoder and alert-rate rule |
| Evaluation | Leave-one-attack-out (LOAO), multiple seeds, recall, precision, F1, false-positive rate and cost |
| Analysis | Held-out attack detectability in embedding space; descriptive score-distribution monitoring in the local API |
| Console | Recorded replay, decision inspection, review queue, cost comparison and contextual MITRE ATT&CK references |

**Stack:** React · TypeScript · Vite · FastAPI · SQLite · PyTorch · scikit-learn. Streamlit remains available as a separate legacy dashboard. ONNX Runtime supports optional CPU inference measurements.

## Results and limitations

The completed synthetic benchmark includes **15 runs: three held-out families × five seeds**.

| Held-out family | Frozen DQN recall, mean ± sample SD | Random Forest recall, mean |
| --- | ---: | ---: |
| Exploits | 100.0% ± 0.0% | 100.0% |
| Fuzzers | 94.7% ± 1.6% | 98.9% |
| Reconnaissance | 48.0% ± 9.4% | 48.7% |

These results do **not** establish DQN superiority. Recall must be considered alongside false alarms and cost. See the [results report](docs/RESEARCH_RESULTS.md) and [raw metric tables](docs/research-results/) for provenance, limitations and model-only CPU timings.

- Preprocessing fits training records only; duplicate feature records cannot cross partitions.
- Held-out families are excluded from initial training and validation. Adaptation uses separate labeled records; adapted results are distinct from frozen results.
- Partitions are custom and replay order is synthetic, not a chronological production stream.
- A first UNSW-NB15 custom held-out-Fuzzers pilot is complete. See [pilot results and provenance](docs/UNSW_PILOT.md). Multi-seed validation and independent lab-traffic validation remain outstanding.
- Analyst labels do not update the running detector. MITRE mappings are contextual references, not model predictions.

## Run locally

### Browser-only demonstration

Requires Node.js 22 or 24. No Python server or credentials are needed.

```sh
git clone https://github.com/Sathvik257/Adaptive-Defense.git
cd Adaptive-Defense
npm ci --prefix web
npm run build:vercel --prefix web
npm run preview --prefix web -- --port 4173
```

Open [localhost:4173](http://127.0.0.1:4173/).

### Full local research console

Requires Python 3.11+ with a compatible PyTorch build, plus Node.js. Create and activate a virtual environment before installing dependencies.

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements-api.txt
npm ci --prefix web
npm run build --prefix web
python serve_api.py
```

Open [localhost:8000](http://127.0.0.1:8000/). This mode uses FastAPI WebSocket replay and a local SQLite review queue. The bundled recorded artifact is ready to explore without retraining.

For the legacy dashboard, run `python -m streamlit run dashboard.py` and open port 8501.

## Reproduce experiments

With the Python environment active:

```sh
python -m pip install .
adaptive-benchmark --demo --output runs/loao-new --seeds 11 22 33 44 55 --epochs 20 --episodes 100
```

Choose a new output directory; completed runs are not overwritten. Outputs include configuration and provenance, split manifests, per-model metrics, predictions, training histories and model artifacts. Only load trusted pickle/model files.

For real data, obtain the prepared CSV files from the [official UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset), then follow the [benchmark guide](docs/BENCHMARK.md) and [dataset card](docs/DATASET_CARD.md). Custom LOAO partitions must not be reported as the official train/test benchmark split.

## Repository structure

```text
adaptive_defense/   Training, evaluation, inference and local API
web/               Original React console and hosted-demo adapter
tests/             Data integrity, models, API and workflow checks
tools/             Artifact export, CPU benchmarking and study utilities
docs/              Methodology, results, cards and deployment guides
lab/               Isolated traffic-generation and capture recipe
paper/             IEEE-style manuscript source
runs/demo/         Bundled synthetic demonstration artifacts
vercel.json        Public web-service routing and hosted build configuration
```

## Validation

```sh
python -m pip install pytest
python -m pytest -q
npm run build --prefix web
npm run build:vercel --prefix web
```

The latest Python check passed 46 tests. Both frontend build modes and browser checks for replay, review persistence, benchmarks, reset and mobile overflow have passed. If Windows denies access to its temporary directory, pass a fresh project-local `--basetemp` to pytest.

## Documentation

| Guide | Contents |
| --- | --- |
| [Project report](docs/PROJECT_REPORT.md) | Academic background and methodology |
| [Benchmark](docs/BENCHMARK.md) | Reproducible LOAO experiments |
| [Dataset card](docs/DATASET_CARD.md) / [Model card](docs/MODEL_CARD.md) | Intended use, splits and limitations |
| [Research results](docs/RESEARCH_RESULTS.md) | Measured synthetic results and remaining work |
| [Related work](docs/RELATED_WORK.md) | Comparison with Kitsune and streaming monitoring |
| [Web console](docs/WEB_CONSOLE.md) | Console architecture and research boundaries |
| [Vercel deployment](docs/VERCEL.md) | Hosted-demo setup and browser storage behavior |
| [Architecture](docs/ARCHITECTURE.md) | Research modules and development workflow |
| [Isolated lab](lab/README.md) | Local traffic-generation protocol |

## Deployment

**Live demo:** [web-flame-eight-75.vercel.app](https://web-flame-eight-75.vercel.app/)

Vercel builds one public `web` service and routes `/` to it. The hosted build publishes static assets and an allowlisted synthetic console artifact; it does not deploy the Python research stack. See [deployment instructions](docs/VERCEL.md) to reproduce the setup.

For the prepared UNSW-NB15 CSV audit and exact experiment commands, see [UNSW-NB15 evaluation](docs/UNSW_NB15.md).
