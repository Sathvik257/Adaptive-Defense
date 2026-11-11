"""Predict with a trusted local ADT checkpoint; feedback is explicitly opt-in."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .adt import ADT, replay
from .adt_experiment import l2_scores
from .data import clean_features
from .models import Autoencoder
from .adt_architecture import ReferenceAutoencoder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True)
    p.add_argument('--csv',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--feedback',action='store_true',help='Reveal each CSV label only AFTER predicting its row.')
    args = p.parse_args()
    root, output = Path(args.run), Path(args.output)
    if output.exists():
        p.error('Output already exists; choose another path.')
    if not (root/'adt_summary.json').is_file():
        p.error('Expected a completed ADT run, not a legacy controller.')
    with (root/'pipeline.pkl').open('rb') as handle:
        bundle = pickle.load(handle)
    if bundle.get('format') not in ('adt-flow-v1','adt-flow-v2'):
        p.error('Unsupported checkpoint format.')
    if not Path(args.csv).is_file():
        p.error('Input CSV does not exist.')
    frame = pd.read_csv(args.csv)
    frame.columns = frame.columns.str.strip().str.lower()
    if frame.columns.duplicated().any() or frame.empty:
        p.error('CSV must contain rows and unique column names.')
    labels = None
    if args.feedback:
        if 'label' not in frame or not frame.label.isin([0,1]).all():
            p.error('--feedback requires binary label values in the CSV.')
        labels = frame.label.to_numpy(dtype=int)
    features = np.asarray(bundle['preprocessor'].transform(clean_features(frame,bundle['columns'])),dtype=np.float32)
    legacy = bundle['format'] == 'adt-flow-v1'
    ae = Autoencoder(bundle['dimension']) if legacy else ReferenceAutoencoder(bundle['dimension'],bundle['latent'])
    ae.load_state_dict(torch.load(root/'autoencoder.pt',map_location='cpu',weights_only=True))
    ae.eval()
    agent = ADT(hidden=(64,64) if legacy else (32,32,16))
    agent.online.load_state_dict(torch.load(root/'adt.pt',map_location='cpu',weights_only=True))
    scores = np.clip(l2_scores(ae,features)/bundle['score_scale'],0,1)
    _, trace = replay(agent,scores,labels,k=bundle['k'],alpha=bundle['alpha'])
    records = pd.DataFrame(trace)
    records['feedback_mode'] = 'previous-label feedback' if args.feedback else 'no-feedback ablation'
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='') as handle:
        records.to_csv(handle,index=False)
    print(f'Saved {len(records)} predictions to {output}')


if __name__ == '__main__':
    main()
