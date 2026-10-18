# Research extension results — 18 October 2026

## Completed

The reusable LOAO benchmark completed 15 synthetic runs: three held-out attack
families × seeds 11, 22, 33, 44, 55. Every run used the same generated dataset
(generator seed 2026), up to 20 autoencoder epochs and 100 DQN episodes.
The dataset source hash, metrics, seed summaries and detectability plot are in
research-results/. Full local models and split manifests are in runs/loao-five-seed/.

| Held-out family | Frozen DQN recall, mean ± sample SD | RF recall, mean | Mean latent distance |
|---|---:|---:|---:|
| Exploits | 100.0% ± 0.0% | 100.0% | 1.389 |
| Fuzzers | 94.7% ± 1.6% | 98.9% | 0.789 |
| Reconnaissance | 48.0% ± 9.4% | 48.7% | 1.000 |

Distance alone does not order detection performance: Reconnaissance is farther
from known attack centroids than Fuzzers on average, yet has lower recall.
This is evidence against a simple monotonic relationship on this synthetic
generator, not a causal explanation or a general law. Geometry differs across
folds. The RF comparison also shows that DQN superiority cannot be claimed.
False-positive rates and all other models are retained in metrics.csv; recall
alone is insufficient to choose a detector.

## Local CPU measurement

ONNX Runtime 1.24.4, CPUExecutionProvider, float32, one intra-op thread,
Windows 11, Intel64 Family 6 Model 186 Stepping 3. Twenty warm-up calls and
200 measured calls per batch. Training and tests had finished before this run.
The machine was not otherwise controlled as a dedicated benchmarking appliance.

| Batch | Median batch latency | p95 batch latency | Flows/s (total work / total inference time) |
|---|---:|---:|---:|
| 1 | 0.0937 ms | 0.1393 ms | 9,925 |
| 32 | 0.2265 ms | 0.2729 ms | 135,470 |
| 256 | 1.1846 ms | 1.4192 ms | 238,188 |

These numbers cover AE reconstruction only, excluding feature extraction,
preprocessing, anomaly-score reduction, DQN and network/API overhead. They are
not end-to-end NIDS capacity or Raspberry Pi measurements. Torch/ONNX maximum
absolute reconstruction difference was 1.79e-7 on the parity sample.

## Verification and remaining work

- 38 Python tests passed; production frontend build passed.
- Version 0.4.0 wheel built and installed; benchmark CLI help executes.
- Browser displays the 15-run plot, measured table and ATT&CK references.
- Isolated lab Compose configuration validates. The Docker Linux engine was
  unavailable, and Docker Desktop was not found at its standard installation
  path. No capture or real-traffic validation has occurred.
- SUS protocol and scorer are ready; five actual participants are still needed.
- IEEEtran paper source includes measured tables and a figure. Preview compilation
  failed twice at the app's Windows sandbox setup, before TeX diagnostics. No PDF
  compilation or venue compliance is claimed. Nothing was submitted to arXiv.
- Code is available through this repository; it has not been uploaded to PyPI
  or split into a separate GitHub repository.
