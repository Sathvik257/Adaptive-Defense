"""Offline prediction with trusted dual-evidence pilot checkpoints."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adaptive_defense.adt_architecture import ReferenceAutoencoder
from adaptive_defense.adt_experiment import l2_scores
from adaptive_defense.data import clean_features
from adaptive_defense.dual_evidence import encode
from adaptive_defense.relevance_controller import RelevanceAgent,controller_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',required=True);p.add_argument('--method-run',required=True)
    p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--feedback',action='store_true',help='Simulate analyst labels delivered 32 rows after a request.')
    args=p.parse_args();torch.set_num_threads(2)
    base,method,out=Path(args.base_run),Path(args.method_run),Path(args.output)
    if out.exists():p.error('Output exists.')
    report=json.loads((method/'report.json').read_text())
    import hashlib
    for name,digest in report['checkpoint_sha256'].items():
        if hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest:raise ValueError('Base checkpoint mismatch.')
    with (base/'pipeline.pkl').open('rb') as f:bundle=pickle.load(f)
    with (method/'dual_evidence.pkl').open('rb') as f:detector=pickle.load(f)
    frame=pd.read_csv(args.csv);frame.columns=frame.columns.str.strip().str.lower()
    if frame.empty or frame.columns.duplicated().any():p.error('Require nonempty CSV with unique columns.')
    if args.feedback and ('label' not in frame or not frame.label.isin([0,1]).all()):p.error('Feedback requires binary labels.')
    ae=ReferenceAutoencoder(bundle['dimension'],bundle['latent'])
    ae.load_state_dict(torch.load(base/'autoencoder.pt',weights_only=True,map_location='cpu'));ae.eval()
    features=np.asarray(bundle['preprocessor'].transform(clean_features(frame,bundle['columns'])),dtype=np.float32)
    evidence=detector.transform(l2_scores(ae,features),encode(ae,features))
    agent=RelevanceAgent();agent.online.load_state_dict(torch.load(method/'relevance_learned.pt',weights_only=True,map_location='cpu'));agent.online.eval()
    _,trace=controller_replay(agent,evidence['rank_pairs'],report['policy_thresholds'],
        feedback=(lambda i:int(frame.label.iloc[i])) if args.feedback else None)
    result=pd.DataFrame(trace)
    result['feedback_mode']='delayed simulated analyst' if args.feedback else 'none'
    result['reconstruction_rank']=evidence['rank_pairs'][:,0]
    result['latent_rank']=evidence['rank_pairs'][:,1]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='') as f:result.to_csv(f,index=False)
    print(f'Saved {len(result)} predictions to {out}')
if __name__=='__main__':main()
