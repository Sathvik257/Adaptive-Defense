import pytest
from fastapi.testclient import TestClient
from adaptive_defense.monitoring import score_shift
from test_console_api import app


def test_shift_and_identical_distributions():
    assert score_shift([0] * 64 + [1] * 64, 128)["ks_distance"] == 1
    assert score_shift(list(range(64)) * 2, 128)["ks_distance"] == 0


def test_warmup_disjoint_and_no_future_leakage():
    assert score_shift([1] * 128, 127)["status"] == "warming-up"
    a = score_shift([0] * 64 + [1] * 64 + [float("nan")], 128)
    b = score_shift([0] * 64 + [1] * 64 + [999], 128)
    assert a == b
    assert a["reference_median"] == 0 and a["recent_median"] == 1


@pytest.mark.parametrize("scores,count,window", [([1],2,64),([1],-1,64),([1],True,64),([1],1,1),([float("nan")],1,64)])
def test_invalid_input(scores,count,window):
    with pytest.raises(ValueError):
        score_shift(scores,count,window)


def test_api_prefix_bounds(app):
    client = TestClient(app)
    assert client.get("/api/monitoring?count=12&window=6").json()["ks_distance"] == 1
    assert client.get("/api/monitoring?count=23").status_code == 422
    assert client.get("/api/monitoring?window=1").status_code == 422
    assert client.get("/api/monitoring").json()["status"] == "warming-up"
