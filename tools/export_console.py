"""Export real model-derived demo observations, with training-only PCA fitting."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from adaptive_defense.data import clean_features
from adaptive_defense.models import Autoencoder
from adaptive_defense.predict import InferencePolicy
from adaptive_defense.environment import ThresholdEnvironment

def export(run=ROOT/'runs/demo'):
    torch.set_num_threads(2)
    with open(run/'pipeline.pkl','rb') as file:
        bundle=pickle.load(file)
    summary=json.loads((run/'summary.json').read_text())
    raw=pd.read_csv(run/'synthetic_input.csv')
    columns=bundle['columns']
    fingerprints=pd.util.hash_pandas_object(raw[columns],index=False).astype(str)
    x=np.asarray(bundle['preprocessor'].transform(clean_features(raw,columns)),dtype=np.float32)
    model=Autoencoder(bundle['dimension'])
    model.load_state_dict(torch.load(run/'autoencoder.pt',weights_only=True))
    model.eval()
    with torch.inference_mode():
        tensor=torch.tensor(x)
        embeddings=model.net[:5](tensor).numpy()
        errors=((model(tensor)-tensor)**2).numpy()
    manifest=pd.read_csv(run/'split_manifest.csv',dtype={'_fingerprint':str})
    train_ids=set(manifest.loc[manifest.split=='train','_fingerprint'])
    training=np.array([key in train_ids for key in fingerprints])
    pca=PCA(n_components=2,random_state=42).fit(embeddings[training])
    projected=pca.transform(embeddings)
    probs=bundle['baselines']['Random Forest'].predict_proba(x)
    entropy=-(probs*np.log2(np.maximum(probs,1e-10))).sum(1)
    predictions=pd.read_csv(run/'predictions.csv',dtype={'_fingerprint':str})
    lookup={value:index for index,value in enumerate(fingerprints)}
    names=bundle['preprocessor'].get_feature_names_out()
    grid=bundle['grid']
    env=ThresholdEnvironment(predictions.anomaly_score.to_numpy(),np.zeros(len(predictions)),grid,summary['configuration']['window'])
    policy=InferencePolicy(run/'dqn.pt')
    state=env.reset(); windows=[]; done=False
    while not done:
        with torch.inference_mode():
            q=policy.online(torch.tensor(state)).numpy().tolist()
        action=policy.action(state)
        state,_,done,info=env.step(action)
        windows.append({**{k:info[k] for k in ('start','end','threshold')},'q_values':q,'threshold_action':['lower','hold','raise'][action]})
    flows=[]
    for position,row in predictions.iterrows():
        index=lookup[row['_fingerprint']]
        window=next(w for w in windows if w['start']<=position<w['end'])
        top=np.argsort(errors[index])[-3:][::-1]
        flows.append({'id':f'flow-{position+1:04d}','index':int(position),
            'score':float(row.anomaly_score),'entropy':float(entropy[index]),
            'x':float(projected[index,0]),'y':float(projected[index,1]),
            'action':'alert' if row['DQN frozen'] else 'allow',
            'threshold':window['threshold'],'threshold_action':window['threshold_action'],
            'q_values':window['q_values'],
            'contributions':[{'feature':str(names[j]).split('__',1)[-1],'error':float(errors[index,j])} for j in top]})
    result={'dataset':'Synthetic demonstration','projection':'PCA of 16-dimensional autoencoder embedding; fitted on training records only',
        'policy':'Saved DQN threshold controller. Analyst queries are user-directed, not PPO decisions.',
        'flows':flows,'metrics':pd.read_csv(run/'metrics.csv').to_dict(orient='records'),
        'summary':summary,'windows':windows,
        'truth':[{'id':f'flow-{i+1:04d}','label':int(row.label),'family':row.attack_cat} for i,row in predictions.iterrows()]}
    (run/'console.json').write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
    print(f'Exported {len(flows)} model-derived flows to {run / "console.json"}')

if __name__=='__main__':
    export()
