"""Predeclared dual-evidence pilot using a frozen, trusted ADT v2 checkpoint."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from adaptive_defense.adt_architecture import ReferenceAutoencoder
from adaptive_defense.adt_experiment import l2_scores
from adaptive_defense.data import split_data, clean_features
from adaptive_defense.dual_evidence import DualEvidence, encode
from adaptive_defense.calibrated_adt import calibrated_threshold
from adaptive_defense.research_diagnostics import require_equal, score_diagnostics
from tools.validate_adt import evaluate
from adaptive_defense.relevance_controller import RelevanceAgent,controller_replay


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--csv',required=True);p.add_argument('--output',required=True)
    p.add_argument('--controller-episodes',type=int,default=0)
    args=p.parse_args();torch.set_num_threads(2)
    if args.controller_episodes<0:p.error('Episodes must be nonnegative.')
    root,source,out=Path(args.run),Path(args.csv),Path(args.output)
    if out.exists():p.error('Choose a fresh output directory.')
    summary=json.loads((root/'adt_summary.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    require_equal([digest],[s['sha256'] for s in summary['sources']],'Source mismatch.')
    splits,columns,audit=split_data(pd.read_csv(source),summary['configuration']['holdout'],summary['configuration']['seed'])
    stored=pd.read_csv(root/'split_manifest.csv',dtype={'_fingerprint':str})
    if not stored._fingerprint.is_unique:raise ValueError('Overlapping source partitions.')
    for name,frame in splits.items():
        require_equal(frame._fingerprint.tolist(),stored.loc[stored.split==name,'_fingerprint'].tolist(),'Partition mismatch.')
    with (root/'pipeline.pkl').open('rb') as f:bundle=pickle.load(f)
    require_equal(bundle['format'],'adt-flow-v2','Expected v2 checkpoint.')
    require_equal(bundle['columns'],columns,'Feature mismatch.')
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ('pipeline.pkl','autoencoder.pt','adt.pt')}
    ae=ReferenceAutoencoder(bundle['dimension'],bundle['latent'])
    ae.load_state_dict(torch.load(root/'autoencoder.pt',weights_only=True,map_location='cpu'));ae.eval()
    def evidence(frame):
        x=np.asarray(bundle['preprocessor'].transform(clean_features(frame,columns)),dtype=np.float32)
        return l2_scores(ae,x),encode(ae,x)
    normal_train=splits['train'].query('label == 0')
    detector=DualEvidence().fit(*evidence(normal_train))
    calibration=splits['adaptation'].query('label == 0')
    cal=detector.transform(*evidence(calibration))
    thresholds={name:calibrated_threshold(cal[name],.05) for name in ('reconstruction','latent_distance','dual_max')}
    controllers={};training_history=[]
    policy_thresholds=np.array([calibrated_threshold(cal['dual_max'],a) for a in (.10,.05,.01)])
    if args.controller_episodes:
        training_pairs=detector.transform(*evidence(splits['train']))['rank_pairs']
        training_y=splits['train'].label.to_numpy(int)
        for name,weighted,mode in [('relevance_learned',True,'learned'),('unweighted_learned',False,'learned'),('relevance_periodic',True,'periodic')]:
            agent=RelevanceAgent(42);rng=np.random.default_rng(42)
            for episode in range(args.controller_episodes):
                selected=rng.choice(len(training_y),256,replace=False)
                target=training_y[selected]
                _,trace=controller_replay(agent,training_pairs[selected],policy_thresholds,
                    feedback=lambda i:int(target[i]),training_labels=target,
                    weighted=weighted,review_mode=mode,epsilon=max(.05,1-episode/max(1,args.controller_episodes*.8)))
                losses=[agent.update(128) for _ in range(4)]
                if (episode+1)%20==0:agent.target.load_state_dict(agent.online.state_dict())
                training_history.append(dict(model=name,episode=episode+1,loss=float(np.mean(losses)),queries=sum(r['queried'] for r in trace)))
            agent.online.eval();controllers[name]=(agent,weighted,mode)
            print('Trained '+name,flush=True)
    rows=[];diagnostics=[];family_rows=[]
    for split in ('validation','test'):
        frame=splits[split];y=frame.label.to_numpy(int)
        scores=detector.transform(*evidence(frame))
        for name,threshold in thresholds.items():
            pred=(scores[name]>threshold).astype(int)
            rows.append(dict(split=split,model=name,alpha=.05,threshold=threshold,**evaluate(y,pred)))
            diagnostics.append(dict(split=split,model=name,**score_diagnostics(y,scores[name])))
            for family in sorted(frame.attack_cat.unique()):
                mask=frame.attack_cat.eq(family).to_numpy()
                family_rows.append(dict(split=split,model=name,family=family,rows=int(mask.sum()),alert_rate=float(pred[mask].mean())))
        for alpha in (.01,.10):
            threshold=calibrated_threshold(cal['dual_max'],alpha)
            pred=(scores['dual_max']>threshold).astype(int)
            rows.append(dict(split=split,model=f'dual_static_{alpha}',alpha=alpha,threshold=threshold,**evaluate(y,pred)))
        for name,(agent,weighted,mode) in controllers.items():
            pred,trace=controller_replay(agent,scores['rank_pairs'],policy_thresholds,
                feedback=lambda i:int(y[i]),weighted=weighted,review_mode=mode)
            rows.append(dict(split=split,model=name,alpha=None,threshold=None,
                queries=sum(r['queried'] for r in trace),delivered=trace[-1]['delivered'],**evaluate(y,pred)))
            for family in sorted(frame.attack_cat.unique()):
                mask=frame.attack_cat.eq(family).to_numpy()
                family_rows.append(dict(split=split,model=name,family=family,rows=int(mask.sum()),alert_rate=float(pred[mask].mean())))
        print('Completed '+split,flush=True)
    require_equal(hashes,{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in hashes},'Checkpoint changed.')
    out.mkdir(parents=True,exist_ok=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'score_diagnostics.csv',index=False)
    pd.DataFrame(family_rows).to_csv(out/'per_family.csv',index=False)
    if training_history:
        pd.DataFrame(training_history).to_csv(out/'controller_training.csv',index=False)
        for name,(agent,_,_) in controllers.items():torch.save(agent.online.state_dict(),out/(name+'.pt'))
    with (out/'dual_evidence.pkl').open('wb') as f:pickle.dump(detector,f)
    report=dict(controller_episodes=args.controller_episodes,controller_seed=42,policy_thresholds=policy_thresholds.tolist(),source_sha256=digest,checkpoint_sha256=hashes,benign_fit_rows=len(normal_train),calibration_rows=len(calibration),
        thresholds=thresholds,protocol='Fixed max of benign-training empirical ranks; alpha=.05; frozen encoder; seed42 Fuzzers holdout.',
        limitations=['Existing validation used for AE checkpoint selection; exploratory development comparison.',
        'Test already examined in prior work; not a fresh confirmation set.',
        'Latent geometry and rank fusion have prior art; no novelty claim.',
        'Only one training seed and held-out family; controller training is a pilot, not established convergence.',
        'Controller uses offline known-family labels for rewards; inference state uses only queried delayed labels.',
        'Controller threshold choices use nominal 10%,5%,1% gates; fixed FP penalty 5 is not a guaranteed FPR constraint.',
        'Relevance is similarity in two score ranks, not full feature or latent-space similarity.'])
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
