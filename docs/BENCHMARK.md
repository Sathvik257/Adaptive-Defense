# Reusable LOAO benchmark

See [measured results and current blockers](RESEARCH_RESULTS.md) for the completed
15-run synthetic evaluation and local CPU timings.

Install from the repository with `python -m pip install .`. The versioned
`adaptive-benchmark` entry point trains every attack-family/seed combination.
No separate service, credential or dataset download is required for the demo.

```powershell
adaptive-benchmark --demo --output runs/loao --seeds 11 22 33 44 55 --epochs 20 --episodes 100
adaptive-benchmark --csv data/raw/train.csv data/raw/test.csv --output runs/real-loao
```

Do not combine official partitions if your research question concerns official
split performance: this command deliberately makes custom splits. See the
dataset and model cards for the exact protocol and limitations.

Artifacts: benchmark.json (source SHA-256, fixed dataset, seeds, protocol),
metrics.csv (per-fold models), summary.csv (mean, sample SD, count),
detectability.csv and detectability.png (frozen analysis), and per-fold
preprocessor/model files, split manifests, predictions and training histories.
The output directory must be new. Partial failures leave inspectable artifacts,
but no completed aggregate; there is not yet an automatic resume mechanism.

Frozen detectability excludes adapted DQN results because its policy has already
seen labels from the withheld family. Distance references contain known training
attacks only. No test points fit centroids, normalization or preprocessing.
The per-record file supports checking whether averaging conceals failure modes.
The summary reports sample SD, not confidence intervals or significance.

For reproducibility, retain raw-source hashes and the environment. Treat pickle
artifacts as trusted local files only. Use a separately pinned environment for
archival research. Model training is CPU-compatible; five seeds may take minutes
or longer depending on dataset size. No cloud jobs or publishing is automatic.

## Other research artifacts

- `lab/`: isolated bounded traffic capture recipe; Docker execution required.
- `tools/benchmark_onnx.py`: model-only CPU latency and Torch/ONNX parity.
- `tools/score_sus.py`: score actual anonymous participant responses.
- `paper/`: IEEE conference-class draft; author details and evidence review needed.
- `docs/DATASET_CARD.md` and `docs/MODEL_CARD.md`: scope and limitations.

ATT&CK links are contextual references, not binary-model predictions. T1046
requires service-enumeration evidence; T1499 concerns endpoint denial of service.
Broad Exploits and Fuzzers classes remain unmapped without additional evidence.

Sources: [MITRE T1046](https://attack.mitre.org/techniques/T1046/),
[MITRE T1499](https://attack.mitre.org/techniques/T1499/),
[Docker internal networks](https://docs.docker.com/reference/compose-file/networks/),
[ONNX thread settings](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).
