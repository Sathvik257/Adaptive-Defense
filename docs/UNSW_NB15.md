# UNSW-NB15 evaluation

Checked 6 November 2026 against https://research.unsw.edu.au/projects/unsw-nb15-dataset . The official download link redirected to Microsoft sign-in during access. Download the prepared training and testing CSVs through the official source and place them in data/raw. These files are excluded from Git. The dataset mixes normal activity with generated attack behavior in a cyber-range environment. Follow the source page's academic-use and citation requirements.

## Audit before training

```powershell
.\.venv\Scripts\python.exe -m adaptive_defense.unsw --train data/raw/UNSW_NB15_training-set.csv --test data/raw/UNSW_NB15_testing-set.csv --output runs/unsw-audit/report.json
```

The audit records hashes, class counts, feature duplication, conflicting targets and cross-file overlap. It rejects missing categories, invalid labels and mismatched schemas. A new report path is required each time. Inspect the report and resolve conflicts before training. Source identity still requires provenance from the download provider.

## First custom LOAO experiment

```powershell
.\.venv\Scripts\python.exe -m adaptive_defense.experiment --csv data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv --holdout Fuzzers --output runs/unsw-fuzzers-42 --seed 42 --epochs 40 --episodes 100 --window 256
```

This command combines the inputs and makes custom partitions. Report it as custom leave-one-attack-out evaluation. The official train/test partition evaluation requires a separate protocol and must not be claimed from this command. Frozen and adapted DQN results must be reported separately. Compare recall, precision, F1, false-positive rate and cost for every model.

## Repeatable benchmark

```powershell
.\.venv\Scripts\python.exe -m adaptive_defense.benchmark --csv data/raw/UNSW_NB15_training-set.csv data/raw/UNSW_NB15_testing-set.csv --output runs/unsw-loao --seeds 11 22 33 44 55 --epochs 40 --episodes 100
```

Check small-family sample counts and available compute before running all folds. The bundled website continues to show the completed synthetic experiment. Publish real-dataset results only after training and evaluation complete and their artifacts are reviewed.

Current status: mirror CSVs were audited and a custom held-out-Fuzzers pilot completed after excluding conflicting feature groups. See [pilot results and reproducible commands](UNSW_PILOT.md).
