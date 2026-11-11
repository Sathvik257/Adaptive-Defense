"""Run the ADT paper's controller on a flow-level UNSW-NB15 adaptation."""
import argparse
import hashlib
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from .adt import ADT, replay
from .data import split_data, clean_features, make_preprocessor, generate_demo
from .models import Autoencoder
from .experiment import metrics


def fit_autoencoder(train, validation, epochs, seed):
    torch.manual_seed(seed)
    model = Autoencoder(train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x = torch.tensor(train, dtype=torch.float32)
    v = torch.tensor(validation, dtype=torch.float32)
    history, best, weights = [], float('inf'), None
    for epoch in range(epochs):
        losses = []
        for idx in torch.randperm(len(x)).split(256):
            loss = torch.linalg.vector_norm(x[idx]-model(x[idx]), dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        with torch.no_grad():
            val = torch.linalg.vector_norm(v-model(v),dim=1).mean().item()
        history.append({'epoch':epoch+1,'training_l2':float(np.mean(losses)),'validation_l2':val})
        if val < best:
            best, weights = val, {k:t.detach().clone() for k,t in model.state_dict().items()}
    model.load_state_dict(weights)
    model.eval()
    return model, history


def l2_scores(model, features):
    with torch.no_grad():
        return np.concatenate([torch.linalg.vector_norm(batch-model(batch),dim=1).numpy()
            for batch in torch.tensor(features,dtype=torch.float32).split(1024)])


def static_threshold(scores, labels):
    # sklearn uses >=; move thresholds down one float for our strict > rule.
    precision, recall, thresholds = precision_recall_curve(labels,scores)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return float(np.nextafter(thresholds[np.argmax(f1)], -np.inf))


def run(args):
    torch.set_num_threads(2)
    if min(args.epochs,args.episodes,args.controller_rows,args.k,args.action_interval,args.target_every)<1:
        raise ValueError('All training sizes and intervals must be positive.')
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    sources = [generate_demo(out/'synthetic_input.csv',args.seed)] if args.demo else args.csv
    frame = pd.concat([pd.read_csv(p) for p in sources],ignore_index=True)
    splits, columns, audit = split_data(frame,args.holdout,args.seed)
    prep = make_preprocessor(splits['train'],columns)
    # Numeric min/max is fitted on initial training only; categorical encoding is unchanged.
    prep.transformers[0][1].set_params(scale=MinMaxScaler(clip=True))
    prep.fit(clean_features(splits['train'],columns))
    x = {k:np.asarray(prep.transform(clean_features(v,columns)),dtype=np.float32) for k,v in splits.items()}
    y = {k:v.label.to_numpy(dtype=int) for k,v in splits.items()}
    print('Training benign-only L2 autoencoder...',flush=True)
    ae, ae_history = fit_autoencoder(x['train'][y['train']==0],x['validation'][y['validation']==0],args.epochs,args.seed)
    raw = {k:l2_scores(ae,v) for k,v in x.items()}
    # Paper requires [0,1] but leaves score normalization unspecified.
    # Freeze an initial-training maximum; never fit scaling to evaluation scores.
    scale = max(float(raw['train'].max()),1e-12)
    scores = {k:np.clip(v/scale,0,1) for k,v in raw.items()}
    threshold = static_threshold(scores['validation'],y['validation'])
    rng = np.random.default_rng(args.seed)
    count = min(args.controller_rows,len(y['train']))
    if count < 2:
        raise ValueError('Controller subset needs at least two rows.')
    # Ensure both classes without using held-out-family or test labels.
    chosen = [int(rng.choice(np.flatnonzero(y['train']==label))) for label in (0,1)]
    rest = np.setdiff1d(np.arange(len(y['train'])),chosen)
    chosen = np.sort(np.r_[chosen,rng.choice(rest,count-2,replace=False)])
    agent, history = ADT(args.seed), []
    print(f'Training ADT on {count} development flows for {args.episodes} episodes...',flush=True)
    for episode in range(args.episodes):
        epsilon = max(.05,1-episode/max(1,args.episodes*.8))
        _, trace = replay(agent,scores['train'][chosen],y['train'][chosen],k=args.k,
            training=True,epsilon=epsilon,action_interval=args.action_interval)
        loss = agent.update(batch_size=min(64,count))  # One gradient update per episode.
        if (episode+1)%args.target_every == 0:
            agent.target.load_state_dict(agent.online.state_dict())
        history.append({'episode':episode+1,'epsilon':epsilon,'loss':loss,
                        'mean_reward':float(np.mean([r['reward'] for r in trace]))})
    predictions = {'Static AE (validation F1)':(scores['test']>threshold).astype(int)}
    predictions['ADT (previous-label feedback)'], trace = replay(agent,scores['test'],y['test'],k=args.k)
    predictions['ADT (no-feedback ablation)'], no_feedback = replay(agent,scores['test'],k=args.k)
    # Exposes how much previous-label persistence alone can explain.
    predictions['Previous-label baseline'] = np.r_[0,y['test'][:-1]]
    results = pd.DataFrame([{'model':name,**metrics(y['test'],pred,splits['test'].attack_cat,
                         audit['held_out_categories'])} for name,pred in predictions.items()])
    results.to_csv(out/'metrics.csv',index=False)
    records = splits['test'][['_fingerprint','label','attack_cat']].copy()
    records['anomaly_score'] = scores['test']
    for name,pred in predictions.items():
        records[name] = pred
    records.to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(trace).to_csv(out/'feedback_trace.csv',index=False)
    pd.DataFrame(no_feedback).to_csv(out/'no_feedback_trace.csv',index=False)
    pd.DataFrame(history).to_csv(out/'controller_history.csv',index=False)
    pd.DataFrame(ae_history).to_csv(out/'autoencoder_history.csv',index=False)
    pd.concat([v[['_fingerprint','label','attack_cat']].assign(split=k) for k,v in splits.items()]).to_csv(out/'split_manifest.csv',index=False)
    splits['train'].iloc[chosen][['_fingerprint','label','attack_cat']].to_csv(out/'controller_subset.csv',index=False)
    torch.save(ae.state_dict(),out/'autoencoder.pt')
    torch.save(agent.online.state_dict(),out/'adt.pt')
    with (out/'pipeline.pkl').open('wb') as handle:
        pickle.dump({'format':'adt-flow-v1','preprocessor':prep,'columns':columns,
                     'dimension':x['train'].shape[1],'score_scale':scale,'static_threshold':threshold,
                     'k':args.k,'alpha':.9},handle)
    summary = {'method':'ADT flow adaptation v1','paper':'https://arxiv.org/abs/2312.01488v1',
        'dataset':'SYNTHETIC DEMONSTRATION' if args.demo else 'USER-SUPPLIED NETWORK FLOW CSV',
        'configuration':vars(args),'audit':audit,'tau':1,'alpha':.9,'beta':.1,
        'controller_training_rows':count,'controller_fraction_of_unique_data':count/audit['unique_records'],
        'score_scale_train_max':scale,'static_threshold_validation_f1':threshold,
        'score_clipping_counts':{k:int((v>scale).sum()) for k,v in raw.items()},
        'sources':[{'name':Path(p).name,'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()} for p in sources],
        'elapsed_seconds':time.perf_counter()-started,
        'versions':{'numpy':np.__version__,'torch':torch.__version__,'pandas':pd.__version__},
        'limitations':['UNSW adaptation, not reproduction on Yahoo/SWaT/WADI.',
            'Randomized flow replay has no verified temporal ordering; tau=1.',
            'Paper-mode evaluation receives each true label AFTER its decision; weights remain frozen.',
            'No-feedback ablation masks the confusion features and is not the paper protocol.',
            'Held-out-family feedback enters later states; this is not wholly label-free zero-day detection.',
            'DSPOT and the paper datasets have not been reproduced.',
            'Network widths, optimizer settings, normalization and startup handling are implementation choices.']}
    # Separate manifest name prevents legacy dashboards from loading incompatible checkpoints.
    (out/'adt_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    html = '<!doctype html><meta charset="utf-8"><title>Adaptive Defense - ADT experiment</title>'
    html += '<style>body{font:16px system-ui;margin:40px auto;max-width:1100px;color:#16372e;background:#f4f7f5}table{border-collapse:collapse;background:white}td,th{padding:12px;border-bottom:1px solid #ddd}h1{color:#126548}</style>'
    html += '<h1>Adaptive Defense / ADT</h1><p>Paper-aligned controller · UNSW flow adaptation · research evaluation</p>'
    html += results.round(4).to_html(index=False)
    html += '<h2>How to read these results</h2><ul>'+''.join('<li>'+s+'</li>' for s in summary['limitations'])+'</ul>'
    (out/'report.html').write_text(html,encoding='utf-8')
    print(results.to_string(index=False),flush=True)
    print(f'Saved {out.resolve()}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo',action='store_true')
    source.add_argument('--csv',nargs='+')
    parser.add_argument('--output',required=True)
    parser.add_argument('--holdout',nargs='+',default=['Fuzzers'])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--episodes',type=int,default=2000)
    parser.add_argument('--controller-rows',type=int,default=256)
    parser.add_argument('--k',type=int,default=2)
    parser.add_argument('--action-interval',type=int,default=1)
    parser.add_argument('--target-every',type=int,default=50)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
