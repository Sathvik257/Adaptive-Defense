import pandas as pd
import pytest
from adaptive_defense.unsw import inspect_pair

def files(tmp_path):
    frame = pd.DataFrame({"label":[0,1],"attack_cat":["Normal","Fuzzers"],"proto":["tcp","udp"],"service":["http","-"],"state":["FIN","INT"],"dur":[1,2],"spkts":[3,4],"dpkts":[5,6]})
    paths = [tmp_path/"train.csv", tmp_path/"test.csv"]
    for path in paths:
        frame.to_csv(path,index=False)
    return frame, paths

def test_overlap_and_conflicts(tmp_path):
    frame, paths = files(tmp_path)
    report = inspect_pair(*paths)
    assert report["cross_file_feature_overlap"] == 2
    assert report["feature_labels_consistent"]
    frame.loc[1,"attack_cat"] = "Exploits"
    frame.to_csv(paths[1],index=False)
    assert inspect_pair(*paths)["conflicting_feature_groups"] == 1

def test_reject_mismatched_labels(tmp_path):
    frame, paths = files(tmp_path)
    frame.loc[0,"attack_cat"] = "Fuzzers"
    frame.to_csv(paths[0],index=False)
    with pytest.raises(ValueError, match="disagree"):
        inspect_pair(*paths)


def test_preparation_excludes_entire_conflict_group(tmp_path, monkeypatch):
    from tools.prepare_unsw import main
    frame, paths = files(tmp_path)
    frame.loc[1, "attack_cat"] = "Exploits"
    frame.to_csv(paths[1], index=False)
    output = tmp_path / "prepared.csv"
    monkeypatch.setattr("sys.argv", ["prepare", "--train", str(paths[0]), "--test", str(paths[1]), "--output", str(output), "--source", "fixture"])
    main()
    prepared = pd.read_csv(output)
    assert len(prepared) == 2
    assert set(prepared.attack_cat) == {"normal"}
    with pytest.raises(ValueError, match="new output"):
        main()
