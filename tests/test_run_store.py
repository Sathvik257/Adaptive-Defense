from pathlib import Path
import pytest
from adaptive_defense.run_store import discover_runs,load_experiment

def test_missing_run_reports_files(tmp_path):
    with pytest.raises(ValueError,match="Incomplete experiment"):
        load_experiment(tmp_path)

def test_demo_artifacts_match():
    run=Path("runs/demo")
    if not run.exists():
        pytest.skip("Demo artifacts not present")
    experiment=load_experiment(run)
    assert experiment.is_demo
    assert len(experiment.metrics)==6
    assert len(experiment.predictions)>0
    assert run in discover_runs("runs")
