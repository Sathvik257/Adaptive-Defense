"""Reload a trusted ADT v2 run and independently evaluate its held-out partitions."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from adaptive_defense.adt import ADT, replay
from adaptive_defense.adt_architecture import ReferenceAutoencoder
from adaptive_defense.adt_experiment import l2_scores
from adaptive_defense.data import split_data, clean_features


def evaluate(y,p):
    tn,fp,fn,tp=map(int,confusion_matrix(y,p,labels=[0,1]).ravel())
    precision,recall,f1,_=precision_recall_fscore_support(y,p,average='binary',zero_division=0)
    return dict(precision=float(precision),recall=float(recall),f1=float(f1),
                accuracy=float(accuracy_score(y,p)),false_positive_rate=fp/max(tn+fp,1),tn=tn,fp=fp,fn=fn,tp=tp)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--csv',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    run,source,out=Path(args.run),Path(args.csv),Path(args.output)
    summary=json.loads((run/'adt_summary.json').read_text())
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    if len(summary['sources'])!=1 or digest!=summary['sources'][0]['sha256']:
        raise ValueError('Source hash differs from the original single-file run.')
    splits,columns,audit=split_data(pd.read_csv(source),summary['configuration']['holdout'],summary['configuration']['seed'])
    stored=pd.read_csv(run/'split_manifest.csv',dtype={'_fingerprint':str})
    if not stored._fingerprint.is_unique:
        raise ValueError('Fingerprint overlap in saved partitions.')
    for name,frame in splits.items():
        expected=stored[stored.split==name].reset_index(drop=True)
        pd.testing.assert_frame_equal(frame[['_fingerprint','attack_cat','label']].reset_index(drop=True),expected[['_fingerprint','attack_cat','label']],check_dtype=False)
    for name in ('train','validation'):
        if splits[name].attack_cat.str.lower().isin(audit['held_out_categories']).any():
            raise ValueError('Held-out family leaked into initial development data.')
    out.mkdir(parents=True,exist_ok=False)
    with (run/'pipeline.pkl').open('rb') as handle:
        bundle=pickle.load(handle)
    if bundle['format']!='adt-flow-v2' or bundle['columns']!=columns:
        raise ValueError('Expected the matching ADT v2 feature pipeline.')
    model=ReferenceAutoencoder(bundle['dimension'],bundle['latent'])
    model.load_state_dict(torch.load(run/'autoencoder.pt',map_location='cpu',weights_only=True));model.eval()
    agent=ADT();agent.online.load_state_dict(torch.load(run/'adt.pt',map_location='cpu',weights_only=True));agent.online.eval()
    checkpoint_hashes={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['autoencoder.pt','adt.pt','pipeline.pkl']}
    results,families=[],[]
    original=pd.read_csv(run/'predictions.csv',dtype={'_fingerprint':str})
    for split in ('validation','test'):
        frame=splits[split];y=frame.label.to_numpy(dtype=int)
        features=np.asarray(bundle['preprocessor'].transform(clean_features(frame,columns)),dtype=np.float32)
        scores=np.clip(l2_scores(model,features)/bundle['score_scale'],0,1)
        predictions={'Static AE (validation F1)':(scores>bundle['static_threshold']).astype(int)}
        predictions['ADT (previous-label feedback)'],_=replay(agent,scores,y,k=bundle['k'],alpha=bundle['alpha'])
        predictions['ADT (no-feedback ablation)'],_=replay(agent,scores,k=bundle['k'],alpha=bundle['alpha'])
        predictions['Previous-label baseline']=np.r_[0,y[:-1]]
        for name,prediction in predictions.items():
            results.append({'split':split,'model':name,'rows':len(y),**evaluate(y,prediction)})
            for family in sorted(frame.attack_cat.unique()):
                mask=frame.attack_cat.eq(family).to_numpy()
                families.append({'split':split,'model':name,'family':family,'rows':int(mask.sum()),
                                 'alert_rate':float(prediction[mask].mean()),
                                 'metric':'false-positive rate' if family.lower()=='normal' else 'attack recall'})
            if split=='test':
                np.testing.assert_array_equal(prediction,original[name].to_numpy())
        if split=='test':
            np.testing.assert_array_equal(frame._fingerprint.to_numpy(),original._fingerprint.to_numpy())
            np.testing.assert_allclose(scores,original.anomaly_score.to_numpy(),rtol=1e-6,atol=1e-7)
        print(f'Validated {split}: {len(y)} flows',flush=True)
    for name,digest in checkpoint_hashes.items():
        assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest
    pd.DataFrame(results).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(families).to_csv(out/'per_family.csv',index=False)
    report={'run':run.name,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'checkpoint_sha256':checkpoint_hashes,'checks':{'source_hash_match':True,'saved_partition_order_match':True,
            'disjoint_feature_fingerprints':True,'heldout_absent_from_train_and_validation':True,
            'reloaded_test_predictions_match':True,'checkpoint_bytes_unchanged':True},
            'limitations':['Validation is a development split used for AE checkpoint and static-threshold selection.',
            'Test is the same held-out split reported at training completion, not a new external dataset.',
            'Feedback mode reveals labels after each prediction; no-feedback mode uses no labels.',
            'Only one seed and one held-out family; randomized flow replay is not chronological.',
            'No synthetic-to-UNSW transfer experiment was run.']}
    (out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(pd.DataFrame(results).to_string(index=False))
    print('All validation checks passed; no fitting or weight updates performed.')


if __name__=='__main__':
    main()
