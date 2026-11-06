# UNSW-NB15 pilot — 6 November 2026

Completed one custom held-out-Fuzzers experiment, seed 42, 10 AE epochs, 10 DQN episodes, window 256 and one adaptation pass. These are pilot settings. The multi-family, multi-seed evaluation remains pending.

Inputs came from https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/tree/3eb02c4 after the official UNSW download redirected to sign-in. The mirror train.csv contains 82,332 rows and test.csv contains 175,341, reversing the counts associated with the official filenames. Inputs were combined for custom LOAO. Their hashes identify the downloaded bytes but have not been matched to publisher-issued checksums.

Audit found 2,445 conflicting feature groups and 1,302 shared feature fingerprints across files. All 40,400 rows in conflicting groups were excluded. Remaining duplicate features were removed by the existing splitter. Final partitions contain 79,490 train, 19,875 validation, 20,747 adaptation and 31,127 test records. Exclusion changes the evaluated population and must accompany any reported results.

| Model | Held-out Fuzzers recall | False-positive rate | Overall F1 |
| --- | ---: | ---: | ---: |
| Fixed autoencoder | 33.98% | 20.93% | 0.6041 |
| Random Forest | 14.46% | 2.20% | 0.6292 |
| Supervised MLP | 24.84% | 3.76% | 0.6750 |
| Rule controller | 3.88% | 1.35% | 0.2204 |
| DQN frozen | 33.63% | 20.48% | 0.6027 |
| DQN adapted | 33.24% | 19.92% | 0.6014 |

Adapted DQN received labels on the separate adaptation partition. Frozen DQN is the relevant unseen-family result. The pilot shows weak held-out recall and high false alarms for the AE-based methods. Longer training and repeated seeds need evaluation before selecting a model. The official test split was not preserved by this experiment.

## Reproduction

With downloaded mirror files in data/raw, run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m tools.prepare_unsw --train data/raw/unsw-mirror-train.csv --test data/raw/unsw-mirror-test.csv --output data/raw/unsw-clean-new.csv --source https://huggingface.co/datasets/Mireu-Lab/UNSW-NB15/tree/3eb02c4
.\.venv\Scripts\python.exe -m adaptive_defense.experiment --csv data/raw/unsw-clean-new.csv --holdout Fuzzers --output runs/unsw-pilot-new --seed 42 --epochs 10 --episodes 10 --window 256 --adapt-passes 1
```

[Metrics](unsw-pilot/metrics.csv) and [provenance](unsw-pilot/provenance.json) accompany this report. Local trained artifacts are under runs/unsw-fuzzers-pilot-42. The website's bundled replay still represents the synthetic demonstration.
