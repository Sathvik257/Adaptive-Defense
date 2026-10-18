"""Verify the demo deployment's artifacts, dependencies and measured inference."""
import importlib.metadata
import json
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main():
    run=ROOT/"runs/demo"
    required=["summary.json","metrics.csv","predictions.csv","threshold_trace.csv",
              "autoencoder_history.csv","controller_history.csv","pipeline.pkl",
              "autoencoder.pt","dqn.pt","dqn_adapted.pt"]
    missing=[name for name in required if not (run/name).is_file()]
    if missing:
        raise RuntimeError(f"Missing deployment artifacts: {missing}")
    versions={name:importlib.metadata.version(name) for name in ["torch","numpy","pandas","scikit-learn","streamlit"]}
    import pandas as pd
    import torch
    torch.set_num_threads(2)
    from adaptive_defense.predict import predict
    frame=pd.read_csv(run/"synthetic_input.csv",nrows=1000).drop(columns=["label","attack_cat"])
    timings={}
    for mode in ["fixed","frozen","adapted"]:
        started=time.perf_counter()
        result=predict(run,frame,mode)
        timings[mode]={"rows":len(result),"seconds_including_model_load":round(time.perf_counter()-started,4)}
        assert len(result)==len(frame)
    report={"checked_at_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "python":sys.version.split()[0],"versions":versions,
        "demo_artifact_bytes":sum(p.stat().st_size for p in run.rglob("*") if p.is_file()),
        "inference":timings,"missing_artifacts":missing,
        "notes":["Measured locally on synthetic data, not a concurrency or Linux hosting benchmark.",
                 "2 vCPU and 2 GB RAM is a starting deployment allocation, not a measured minimum."]}
    try:
        import psutil
        report["process_rss_bytes_after_inference"]=psutil.Process().memory_info().rss
    except ImportError:
        report["process_rss_bytes_after_inference"]=None
    destination=ROOT/"docs/resource-check.json"
    destination.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
