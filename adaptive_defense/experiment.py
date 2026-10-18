"""CLI: prepare data, train models, evaluate and save reproducible artifacts."""
import argparse
import copy
import json
import pickle
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from .data import generate_demo, split_data, clean_features, make_preprocessor
from .models import train_autoencoder, anomaly_scores, DQN
from .environment import rollout

def metrics(y,p,categories,holdout):
    tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel()
    unseen=np.isin(np.char.lower(np.array(categories,dtype=str)),holdout)
    known=(np.asarray(y)==1)&~unseen
    return {"precision":float(precision_score(y,p,zero_division=0)),
        "recall":float(recall_score(y,p,zero_division=0)),"f1":float(f1_score(y,p,zero_division=0)),
        "false_positive_rate":float(fp/max(tn+fp,1)),
        "unseen_recall":float(np.mean(p[unseen])) if unseen.any() else None,
        "known_recall":float(np.mean(p[known])) if known.any() else None,
        "tn":int(tn),"fp":int(fp),"fn":int(fn),"tp":int(tp)}

def run(args):
    torch.set_num_threads(2)
    np.random.seed(args.seed)
    out=Path(args.output)
    if (out/"summary.json").exists():
        raise ValueError("Output already contains a run. Choose another --output directory.")
    out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    if args.demo:
        sources=[generate_demo(out/"synthetic_input.csv",args.seed)]
    else:
        sources=args.csv
        if not sources:
            raise ValueError("Supply --csv files or --demo.")
    df=pd.concat([pd.read_csv(p) for p in sources],ignore_index=True)
    splits,columns,audit=split_data(df,args.holdout,args.seed)
    print("Partitions:",audit["split_counts"],flush=True)
    prep=make_preprocessor(splits["train"],columns)
    prep.fit(clean_features(splits["train"],columns))
    x={k:np.asarray(prep.transform(clean_features(v,columns)),dtype=np.float32) for k,v in splits.items()}
    y={k:v.label.to_numpy(dtype=int) for k,v in splits.items()}
    if not all(np.isfinite(v).all() for v in x.values()):
        raise ValueError("Non-finite processed values.")
    model,history=train_autoencoder(x["train"][y["train"]==0],x["validation"][y["validation"]==0],args.epochs,args.seed)
    scores={k:anomaly_scores(model,v) for k,v in x.items()}
    grid=np.quantile(scores["validation"][y["validation"]==0],np.linspace(.80,.999,21))
    # Tune the fixed baseline using development labels only.
    costs=[np.mean(5*((scores["validation"]<=t)&(y["validation"]==1))+
                   ((scores["validation"]>t)&(y["validation"]==0))) for t in grid]
    threshold=float(grid[np.argmin(costs)])
    print("Autoencoder trained; training controller and baselines...",flush=True)
    agent=DQN(args.seed)
    rng=np.random.default_rng(args.seed)
    training=[]
    best_cost=float("inf"); best_weights=None
    for episode in range(args.episodes):
        order=rng.permutation(len(y["train"]))
        # Alternate feedback availability to support label-free evaluation.
        _,trace=rollout(agent,scores["train"][order],y["train"][order],grid,args.window,
                        train=True,epsilon=max(.05,1-episode/max(args.episodes*.8,1)),feedback=episode%2==0)
        vp,_=rollout(agent,scores["validation"],y["validation"],grid,args.window)
        cost=float(np.mean(5*((vp==0)&(y["validation"]==1))+((vp==1)&(y["validation"]==0))))
        training.append({"episode":episode+1,"reward":float(np.mean([t["reward"] for t in trace])),"validation_cost":cost})
        if cost<best_cost:
            best_cost=cost
            best_weights=copy.deepcopy(agent.online.state_dict())
    agent.online.load_state_dict(best_weights)
    agent.target.load_state_dict(best_weights)
    baseline_models={
        "Random Forest":RandomForestClassifier(n_estimators=100,max_depth=18,class_weight="balanced",random_state=args.seed,n_jobs=2),
        "Supervised MLP":MLPClassifier(hidden_layer_sizes=(64,32),max_iter=150,early_stopping=True,random_state=args.seed)}
    predictions={"Fixed autoencoder":(scores["test"]>threshold).astype(int)}
    for name,estimator in baseline_models.items():
        estimator.fit(x["train"],y["train"])
        predictions[name]=estimator.predict(x["test"])
    predictions["Rule controller"],_=rollout(None,scores["test"],y["test"],grid,args.window,rule=True)
    predictions["DQN frozen"],trace=rollout(agent,scores["test"],y["test"],grid,args.window)
    # Clone policy; adaptation never consumes final evaluation records.
    adapted=copy.deepcopy(agent)
    adapted.optimizer=torch.optim.Adam(adapted.online.parameters(),lr=3e-4)
    for _ in range(args.adapt_passes):
        rollout(adapted,scores["adaptation"],y["adaptation"],grid,args.window,train=True,epsilon=.05,feedback=True)
    predictions["DQN adapted"],adapt_trace=rollout(adapted,scores["test"],y["test"],grid,args.window)
    rows=[]
    holdout=audit["held_out_categories"]
    for name,pred in predictions.items():
        rows.append({"model":name,**metrics(y["test"],pred,splits["test"].attack_cat,holdout)})
    results=pd.DataFrame(rows)
    results.to_csv(out/"metrics.csv",index=False)
    records=splits["test"][["_fingerprint","attack_cat","label"]].copy()
    records["anomaly_score"]=scores["test"]
    for name,pred in predictions.items():
        records[name]=pred
    records.to_csv(out/"predictions.csv",index=False)
    pd.DataFrame(trace).to_csv(out/"threshold_trace.csv",index=False)
    pd.DataFrame(adapt_trace).to_csv(out/"adapted_threshold_trace.csv",index=False)
    pd.DataFrame(history).to_csv(out/"autoencoder_history.csv",index=False)
    pd.DataFrame(training).to_csv(out/"controller_history.csv",index=False)
    manifest=pd.concat([v[["_fingerprint","attack_cat","label"]].assign(split=k) for k,v in splits.items()])
    manifest.to_csv(out/"split_manifest.csv",index=False)
    bundle={"preprocessor":prep,"columns":columns,"dimension":x["train"].shape[1],
            "threshold":threshold,"grid":grid,"baselines":baseline_models,"demo":args.demo}
    with open(out/"pipeline.pkl","wb") as f:
        pickle.dump(bundle,f)
    torch.save(model.state_dict(),out/"autoencoder.pt")
    torch.save(agent.online.state_dict(),out/"dqn.pt")
    torch.save(adapted.online.state_dict(),out/"dqn_adapted.pt")
    import sklearn
    summary={"dataset":"SYNTHETIC DEMONSTRATION" if args.demo else "USER-SUPPLIED NETWORK FLOW CSV",
        "seed":args.seed,"configuration":vars(args),"audit":audit,"feature_columns":columns,
        "input_files":[str(p) for p in sources],"elapsed_seconds":time.perf_counter()-started,
        "versions":{"python":platform.python_version(),"torch":torch.__version__,"sklearn":sklearn.__version__,"pandas":pd.__version__},
        "limitations":["Synthetic replay is not chronological traffic.","Withheld categories are a proxy, not verified real zero-day exploits.",
                        "Adapted-category recall is post-feedback performance, not unseen-category generalization.",
                        "DQN superiority is not assumed; compare the saved measurements."]}
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(results.to_string(index=False),flush=True)
    print(f"Saved complete run to {out.resolve()}",flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo",action="store_true")
    p.add_argument("--csv",nargs="+")
    p.add_argument("--output",default="runs/demo")
    p.add_argument("--holdout",nargs="+",default=["Fuzzers"])
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--epochs",type=int,default=40)
    p.add_argument("--episodes",type=int,default=100)
    p.add_argument("--window",type=int,default=256)
    p.add_argument("--adapt-passes",type=int,default=3)
    args=p.parse_args()
    if min(args.epochs,args.episodes,args.window,args.adapt_passes)<1:
        p.error("Epochs, episodes, window and adaptation passes must be positive.")
    if args.demo and args.csv:
        p.error("Use either --demo or --csv, not both.")
    run(args)

if __name__=="__main__":
    main()
