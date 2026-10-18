# Dataset card

## Bundled synthetic data
The generator in adaptive_defense/data.py samples traffic-like tabular features
and shifted distributions for Normal, Exploits, Fuzzers and Reconnaissance.
These names describe artificial classes, not executed attacks. No personal data
or captured network packets occur in this demonstration. Generator seed, source
SHA-256 and experiment seeds are recorded by the benchmark.

## User-supplied data
CSV requires binary label and attack_cat plus compatible flow features. Dataset
licenses and source citations must be supplied by the researcher; this package
does not redistribute UNSW-NB15. Do not infer a dataset license from this code.
Report collection dates, sensor/exporter version, network context, consent,
feature units, missing values, family counts and labeling evidence.

## Partitions
Identical feature rows are deduplicated before partitioning; conflicting binary
labels or attack families are rejected. Known classes use 60/15/10/15 percent train/validation/
adaptation/test. The held-out family uses 40/60 adaptation/test. No held-out
family is present in initial training or validation. The benchmark reports
adapted results separately and excludes them from zero-shot detectability.
Feature preprocessing fits training records only; AE fits benign training data.

These are custom category-stratified partitions, not official or temporal splits.
Deduplication is not host/session independence. Dataset prevalence is retained
within each partition by this allocation, but held-out allocation changes test
prevalence. Report the saved counts; never call these deployment prevalence.

## Limitations
Synthetic separability, feature selection, correlated sessions and collection
artifacts can inflate results. External lab CSVs require exporter compatibility;
matching column names alone does not establish equivalent feature semantics.
