"""Reproducible leave-one-attack-family-out runner and detectability analysis."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from .data import validate, clean_features, generate_demo
from .experiment import run
from .models import Autoencoder


def centroid_distance(train, categories, query):
    """Train-standardized Euclidean distance to nearest known attack centroid.

    Benign records are intentionally excluded from the reference class set.
    Dividing by sqrt(d) produces RMS standardized-coordinate distance.
    """
    train, query = np.asarray(train), np.asarray(query)
    categories = np.asarray(categories)
    if train.ndim != 2 or query.ndim != 2 or not len(train):
        raise ValueError('Nonempty two-dimensional reference embeddings required.')
    scale = np.maximum(train.std(axis=0), 1e-6)
    centers = np.stack([train[categories == c].mean(axis=0) for c in sorted(set(categories))])
    distances = np.sqrt(np.mean(((query[:, None, :] - centers[None, :, :])/scale)**2, axis=2))
    return distances.min(axis=1)


def analyze(run_dir, frame, family, seed):
    """Only trusted locally generated artifacts may be deserialized."""
    run_dir = Path(run_dir)
    with (run_dir/'pipeline.pkl').open('rb') as f:
        bundle = pickle.load(f)
    ids = pd.util.hash_pandas_object(frame[bundle['columns']], index=False).astype(str)
    unique = frame.assign(_fingerprint=ids).drop_duplicates('_fingerprint').set_index('_fingerprint')
    manifest = pd.read_csv(run_dir/'split_manifest.csv', dtype={'_fingerprint':str})
    train_rows = manifest[(manifest.split == 'train') & (manifest.label == 1)]
    test_rows = manifest[(manifest.split == 'test') & (manifest.attack_cat.str.lower() == family.lower())]
    ordered = pd.concat([unique.loc[train_rows._fingerprint], unique.loc[test_rows._fingerprint]])
    x = np.asarray(bundle['preprocessor'].transform(clean_features(ordered, bundle['columns'])), dtype=np.float32)
    model = Autoencoder(bundle['dimension'])
    model.load_state_dict(torch.load(run_dir/'autoencoder.pt', weights_only=True)); model.eval()
    with torch.inference_mode():
        latent = model.net[:5](torch.tensor(x)).numpy()
    distances = centroid_distance(latent[:len(train_rows)], train_rows.attack_cat,
                                  latent[len(train_rows):])
    predictions = pd.read_csv(run_dir/'predictions.csv', dtype={'_fingerprint':str}).set_index('_fingerprint')
    held = predictions.loc[test_rows._fingerprint]
    rows = []
    for name in ['Fixed autoencoder','Random Forest','Supervised MLP','Rule controller','DQN frozen']:
        rows.append({'family':family,'seed':seed,'model':name,'heldout_records':len(held),
                     'distance_mean':float(distances.mean()),'distance_median':float(np.median(distances)),
                     'heldout_recall':float(held[name].mean())})
    pd.DataFrame({'fingerprint':test_rows._fingerprint.to_numpy(),'distance':distances,
                  'detected':held['DQN frozen'].to_numpy()}).to_csv(run_dir/'detectability_records.csv',index=False)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--csv', nargs='+')
    source.add_argument('--demo', action='store_true')
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[11,22,33,44,55])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--episodes', type=int, default=100)
    args = p.parse_args()
    if min(args.epochs,args.episodes)<1 or len(set(args.seeds))!=len(args.seeds):
        p.error('Positive training counts and unique seeds are required.')
    out=Path(args.output)
    if out.exists():
        p.error('Choose a new output directory; benchmark artifacts are immutable.')
    out.mkdir(parents=True)
    sources=args.csv or [str(generate_demo(out/'synthetic.csv', seed=2026))]
    frame=validate(pd.concat([pd.read_csv(path) for path in sources],ignore_index=True))
    families=sorted(frame.loc[frame.label==1,'attack_cat'].unique())
    if len(families)<2:
        p.error('At least two attack families are required.')
    config={'dataset_mode':'synthetic' if args.demo else 'user-supplied',
            'seeds':args.seeds,'families':families,'epochs':args.epochs,'episodes':args.episodes,
            'sources':[{'file':str(path),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in sources],
            'protocol':'LOAO, category-stratified deduplicated splits; not chronological; heldout adaptation rows excluded from frozen analysis'}
    (out/'benchmark.json').write_text(json.dumps(config,indent=2))
    results=[]; detectability=[]
    for index,family in enumerate(families):
        for seed in args.seeds:
            destination=out/f'family-{index:02d}-seed-{seed}'
            run(SimpleNamespace(demo=False,synthetic_source=args.demo,csv=sources,output=str(destination),holdout=[family],
                                seed=seed,epochs=args.epochs,episodes=args.episodes,window=64,adapt_passes=3))
            if args.demo:
                summary_path=destination/'summary.json'
                summary=json.loads(summary_path.read_text())
                summary['dataset']='SYNTHETIC DEMONSTRATION (fixed dataset across seeds)'
                summary['configuration']['dataset_mode']='synthetic'
                summary_path.write_text(json.dumps(summary,indent=2))
            results.append(pd.read_csv(destination/'metrics.csv').assign(family=family,seed=seed))
            detectability.extend(analyze(destination,frame,family,seed))
    metrics=pd.concat(results,ignore_index=True)
    metrics.to_csv(out/'metrics.csv',index=False)
    metrics.groupby(['family','model'])[['unseen_recall','false_positive_rate','f1']].agg(['mean','std','count']).to_csv(out/'summary.csv')
    detail=pd.DataFrame(detectability); detail.to_csv(out/'detectability.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    frozen=detail[detail.model=='DQN frozen']
    for family,group in frozen.groupby('family'):
        ax.scatter(group.distance_mean,group.heldout_recall,label=family,alpha=.75)
    ax.set(xlabel='Mean distance to nearest known attack centroid (standardized RMS)',
           ylabel='Held-out recall · frozen DQN',ylim=(-.03,1.03),
           title='Detectability analysis'+(' · synthetic demonstration' if args.demo else ''))
    ax.legend();ax.grid(alpha=.2);fig.savefig(out/'detectability.png',dpi=180);plt.close(fig)
    print(f'Benchmark complete: {out.resolve()}')


if __name__=='__main__':
    main()
