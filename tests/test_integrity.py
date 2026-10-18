import numpy as np
import pandas as pd
import pytest
from adaptive_defense.data import generate_demo,split_data,clean_features,make_preprocessor
from adaptive_defense.environment import ThresholdEnvironment,rollout
from adaptive_defense.experiment import metrics

@pytest.fixture
def dataset(tmp_path):
    return pd.read_csv(generate_demo(tmp_path/"demo.csv"))

def test_holdout_and_fingerprints_are_isolated(dataset):
    splits,columns,audit=split_data(dataset,["Fuzzers"],42)
    assert not {"id","label","attack_cat"}&set(columns)
    assert "Fuzzers" not in set(splits["train"].attack_cat)
    assert "Fuzzers" not in set(splits["validation"].attack_cat)
    assert "Fuzzers" in set(splits["adaptation"].attack_cat)
    assert "Fuzzers" in set(splits["test"].attack_cat)
    seen=set()
    for frame in splits.values():
        ids=set(frame._fingerprint)
        assert not seen&ids
        seen|=ids

def test_duplicates_cannot_cross_partitions(dataset):
    joined=pd.concat([dataset,dataset.iloc[:50]])
    _,_,audit=split_data(joined,["Fuzzers"],42)
    assert audit["duplicates_removed"]==50

def test_conflicting_duplicate_rejected(dataset):
    duplicate=dataset.iloc[[0]].copy()
    duplicate["label"]=1-duplicate.label
    duplicate["attack_cat"]="Exploits" if duplicate.label.iloc[0] else "Normal"
    with pytest.raises(ValueError,match="conflicting"):
        split_data(pd.concat([dataset,duplicate]),["Fuzzers"],42)

def test_invalid_holdout_rejected(dataset):
    with pytest.raises(ValueError,match="Unknown"):
        split_data(dataset,["Imaginary"],42)

def test_preprocessor_accepts_unseen_category(dataset):
    splits,cols,_=split_data(dataset,["Fuzzers"],42)
    prep=make_preprocessor(splits["train"],cols)
    prep.fit(clean_features(splits["train"],cols))
    test=splits["test"].copy()
    test["proto"]="previously-unseen"
    transformed=prep.transform(clean_features(test,cols))
    assert np.isfinite(transformed).all()

def test_current_labels_cannot_change_initial_observation():
    a=ThresholdEnvironment([.1,.9],[0,1],[.2,.5,.8],window=1,feedback=True)
    b=ThresholdEnvironment([.1,.9],[1,0],[.2,.5,.8],window=1,feedback=True)
    np.testing.assert_array_equal(a.reset(),b.reset())

def test_frozen_policy_observation_never_reveals_labels():
    a=ThresholdEnvironment([.1,.9],[0,1],[.2,.5,.8],window=1)
    b=ThresholdEnvironment([.1,.9],[1,0],[.2,.5,.8],window=1)
    for action in [0,2]:
        state_a,_,_,info_a=a.step(action)
        state_b,_,_,info_b=b.step(action)
        np.testing.assert_array_equal(state_a,state_b)
        np.testing.assert_array_equal(info_a["predictions"],info_b["predictions"])

def test_reward_and_terminal_boundaries():
    env=ThresholdEnvironment([.1,.9],[1,0],[.2,.5,.8],window=2,feedback=True)
    state,reward,done,info=env.step(1)
    assert reward==-3
    assert done
    assert state[5]==.5 and state[6]==.5
    with pytest.raises(RuntimeError):
        env.step(1)

def test_metrics_use_benign_denominator():
    m=metrics(np.array([0,0,1,1]),np.array([0,1,1,0]),["Normal","Normal","Exploits","Fuzzers"],["fuzzers"])
    assert m["false_positive_rate"]==.5
    assert m["unseen_recall"]==0
    assert m["known_recall"]==1
