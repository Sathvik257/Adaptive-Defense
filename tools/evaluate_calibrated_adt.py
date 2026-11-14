"""Evaluate calibrated, budgeted ADT without refitting or test-label selection."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adaptive_defense.adt import ADT, replay
from adaptive_defense.adt_architecture import ReferenceAutoencoder
from adaptive_defense.adt_experiment import l2_scores
from adaptive_defense.data import split_data, clean_features
from adaptive_defense.calibrated_adt import calibrated_threshold, gated_replay
from tools.validate_adt import evaluate
from adaptive_defense.score_monitor import monitor_scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--focused', action='store_true', help='Compare review strategies at fixed alpha=.05 and budget=.05.')
    args = p.parse_args()
    torch.set_num_threads(2)
    root, source, out = Path(args.run), Path(args.csv), Path(args.output)
    summary = json.loads((root/'adt_summary.json').read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != summary['sources'][0]['sha256']:
        raise ValueError('Dataset hash mismatch.')
    splits, columns, audit = split_data(pd.read_csv(source), summary['configuration']['holdout'], summary['configuration']['seed'])
    manifest = pd.read_csv(root/'split_manifest.csv', dtype={'_fingerprint':str})
    if not manifest._fingerprint.is_unique:
        raise ValueError('Overlapping fingerprints.')
    for name, frame in splits.items():
        assert frame._fingerprint.tolist() == manifest.loc[manifest.split == name, '_fingerprint'].tolist()
    with (root/'pipeline.pkl').open('rb') as f:
        bundle = pickle.load(f)
    if bundle['format'] != 'adt-flow-v2' or columns != bundle['columns']:
        raise ValueError('Expected matching v2 pipeline.')
    hashes = {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('pipeline.pkl','autoencoder.pt','adt.pt')}
    ae = ReferenceAutoencoder(bundle['dimension'], bundle['latent'])
    ae.load_state_dict(torch.load(root/'autoencoder.pt', weights_only=True, map_location='cpu')); ae.eval()
    agent = ADT(); agent.online.load_state_dict(torch.load(root/'adt.pt', weights_only=True, map_location='cpu')); agent.online.eval()
    def scores(frame):
        x = np.asarray(bundle['preprocessor'].transform(clean_features(frame, columns)), dtype=np.float32)
        return l2_scores(ae, x) / bundle['score_scale']
    # Previously unused adaptation normals; attacks never enter calibration.
    calibration = splits['adaptation'].query('label == 0')
    cal = scores(calibration)
    test = splits['test']; raw = scores(test); y = test.label.to_numpy(int)
    held = test.attack_cat.str.lower().isin(audit['held_out_categories']).to_numpy()
    rows, families = [], []
    def record(name, pred, alpha=None, budget=None, delay=None, queries=0, delivered=0, strategy="none"):
        rows.append(dict(model=name, strategy=strategy, alpha=alpha, budget=budget, delay=delay, queries=queries,
                         delivered=delivered, heldout_recall=float(pred[held].mean()), **evaluate(y, pred)))
        for family in sorted(test.attack_cat.unique()):
            mask = test.attack_cat.eq(family).to_numpy()
            families.append(dict(model=name,strategy=strategy,alpha=alpha,budget=budget,delay=delay,
                                 family=family,rows=int(mask.sum()),alert_rate=float(pred[mask].mean())))
    record('Static AE validation-F1', (np.clip(raw,0,1) > bundle['static_threshold']).astype(int))
    pred,_ = replay(agent,np.clip(raw,0,1),y,k=bundle['k'],alpha=bundle['alpha']); record('Original ADT full feedback',pred,budget=1.,delay=1,queries=len(y),delivered=len(y)-1)
    thresholds = {}
    for alpha in ((.05,) if args.focused else (.01,.05,.10)):
        threshold = calibrated_threshold(cal,alpha); thresholds[str(alpha)] = threshold
        record('Calibrated AE only', (raw>threshold).astype(int),alpha)
        for budget,delay in (((.05,32),) if args.focused else ((0.,32),(.05,32),(.10,32),(1.,1))):
            for strategy in ('periodic','boundary'):
                pred,trace = gated_replay(agent,raw,threshold,feedback=(lambda i:int(y[i])) if budget else None,
                                        budget=budget,delay=delay,k=bundle['k'],review_strategy=strategy)
                record('Calibrated budgeted ADT',pred,alpha,budget,delay,
                       sum(r['queried'] for r in trace),trace[-1]['delivered'],strategy)
    assert hashes == {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in hashes}
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(families).to_csv(out/'per_family.csv',index=False)
    pd.DataFrame(monitor_scores(cal,raw)).to_csv(out/'score_shift.csv',index=False)
    report = dict(source_sha256=digest,checkpoint_sha256=hashes,calibration_rows=len(cal),test_rows=len(test),thresholds=thresholds,
        limitations=['Score-shift warnings compare mixed test traffic to benign reference; not evidence of temporal drift.',
        'Boundary proximity is a review heuristic, not calibrated model uncertainty.',
        'One seed and held-out Fuzzers family; test already examined in prior research.',
        'Frozen policy trained for original state; review-history state is a deployment ablation, not retrained RL.',
        'Randomized flow replay; delay is rows, not real time. Labels are simulated analyst answers.',
        'Marginal calibration interpretation requires exchangeable benign calibration/test scores; not drift-safe.',
        'No novelty or superiority established; compare calibrated AE to isolate RL contribution.'])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    main()
