"""Score compatible CSV files using a saved, trusted local experiment."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import clean_features
from .models import Autoencoder,QNetwork,anomaly_scores
from .environment import rollout

class InferencePolicy:
    """Read-only Q-policy: no optimizer, replay memory or training initialization."""
    def __init__(self,weights):
        self.online=QNetwork()
        self.online.load_state_dict(torch.load(weights,weights_only=True))
        self.online.eval()

    def action(self,state,epsilon=0):
        with torch.inference_mode():
            return int(self.online(torch.tensor(state,dtype=torch.float32)).argmax().item())

def predict(run,frame,mode="fixed"):
    run=Path(run)
    frame=frame.copy()
    frame.columns=frame.columns.str.strip().str.lower()
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("CSV must contain records and unique column names.")
    with open(run/"pipeline.pkl","rb") as f:
        bundle=pickle.load(f)
    summary=json.loads((run/"summary.json").read_text())
    x=np.asarray(bundle["preprocessor"].transform(clean_features(frame,bundle["columns"])),dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Features produced non-finite values.")
    model=Autoencoder(bundle["dimension"])
    model.load_state_dict(torch.load(run/"autoencoder.pt",weights_only=True))
    scores=anomaly_scores(model,x)
    if mode=="fixed":
        alerts=(scores>bundle["threshold"]).astype(int)
    elif mode in {"frozen","adapted"}:
        agent=InferencePolicy(run/("dqn_adapted.pt" if mode=="adapted" else "dqn.pt"))
        alerts,_=rollout(agent,scores,np.zeros(len(scores),dtype=int),bundle["grid"],summary["configuration"]["window"])
    else:
        raise ValueError("Mode must be fixed, frozen or adapted.")
    return pd.DataFrame({"anomaly_score":scores,"prediction":np.where(alerts,"Suspicious","Normal")})

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--csv",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--mode",choices=["fixed","frozen","adapted"],default="fixed")
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():
        parser.error("Output exists; choose a new filename.")
    result=predict(args.run,pd.read_csv(args.csv),args.mode)
    output.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(output,index=False)
    print(f"Saved {len(result)} predictions to {output}")

if __name__=="__main__":
    main()
