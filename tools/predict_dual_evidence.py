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
from adaptive_defense.explanations import residual_explanations
from adaptive_defense.relevance_controller import RelevanceAgent,controller_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run',required=True);p.add_argument('--method-run',required=True)
    p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--feedback',action='store_true',help='Simulate analyst labels delivered 32 rows after a request.')
    p.add_argument('--detector',choices=['relevance','dual-static'],default='relevance')
    args=p.parse_args();torch.set_num_threads(2)
    if args.detector=='dual-static' and args.feedback:p.error('Static detector does not request feedback.')
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
    if args.detector=='relevance':
        agent=RelevanceAgent();agent.online.load_state_dict(torch.load(method/'relevance_learned.pt',weights_only=True,map_location='cpu'));agent.online.eval()
        _,trace=controller_replay(agent,evidence['rank_pairs'],report['policy_thresholds'],
            feedback=(lambda i:int(frame.label.iloc[i])) if args.feedback else None)
        result=pd.DataFrame(trace)
        result['threshold']=[report['policy_thresholds'][a//2] for a in result.action]
    else:
        threshold=report['thresholds']['dual_max']
        result=pd.DataFrame({'index':np.arange(len(frame)),'prediction':(evidence['dual_max']>threshold).astype(int),
                             'queried':False,'delivered':0,'threshold':threshold})
    explanations=residual_explanations(ae,features,bundle['preprocessor'].get_feature_names_out(),top_k=min(3,features.shape[1]))
    result['detector']=args.detector
    result['raw_reconstruction_l2']=evidence['reconstruction']
    result['latent_mahalanobis_squared']=evidence['latent_distance']
    result['dual_score']=evidence['dual_max']
    result['decision_margin']=evidence['dual_max']-result['threshold']
    result['dominant_evidence']=np.where(evidence['rank_pairs'][:,0]>evidence['rank_pairs'][:,1],'reconstruction',
        np.where(evidence['rank_pairs'][:,0]<evidence['rank_pairs'][:,1],'latent_distance','tie'))
    result['top_reconstruction_features']=[json.dumps(r['top_features'],separators=(',',':')) for r in explanations]
    result['feedback_mode']='delayed simulated analyst' if args.feedback else 'none'
    result['reconstruction_rank']=evidence['rank_pairs'][:,0]
    result['latent_rank']=evidence['rank_pairs'][:,1]
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='') as f:result.to_csv(f,index=False)
    print(f'Saved {len(result)} predictions to {out}')
if __name__=='__main__':main()
