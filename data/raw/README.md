# Real dataset inputs

Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` from the [official UNSW dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and place them here. The official SharePoint link redirected to authentication when checked during implementation; no real dataset is bundled or silently substituted.

Only use authorized copies. Follow the dataset's academic-use and citation requirements. The project combines supplied CSVs and constructs its own documented category-stratified partitions. Therefore its results are **not an official UNSW training/test-split benchmark**.

Expected metadata: `label` (0/1), `attack_cat`. The prepared CSVs normally include categorical `proto`, `service`, `state` columns and numerical flow features. Targets and record IDs are excluded automatically. Other datasets require a reviewed schema adapter; arbitrary CSVs are not guaranteed compatible.
