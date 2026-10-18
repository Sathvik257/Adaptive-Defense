from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from adaptive_defense.predict import predict

@pytest.mark.parametrize("mode",["fixed","frozen","adapted"])
def test_saved_inference_does_not_use_labels(mode):
    if not Path("runs/demo/summary.json").exists():
        pytest.skip("Run demo first")
    frame=pd.read_csv("runs/demo/synthetic_input.csv").head(75)
    labeled=predict("runs/demo",frame,mode)
    unlabeled=predict("runs/demo",frame.drop(columns=["label","attack_cat"]),mode)
    np.testing.assert_array_equal(labeled.prediction,unlabeled.prediction)
    np.testing.assert_allclose(labeled.anomaly_score,unlabeled.anomaly_score)

def test_missing_feature_fails_clearly():
    if not Path("runs/demo/summary.json").exists():
        pytest.skip("Run demo first")
    with pytest.raises(ValueError,match="Missing feature"):
        predict("runs/demo",pd.DataFrame({"wrong":[1]}))
