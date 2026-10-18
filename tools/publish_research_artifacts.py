"""Prepare measured, non-model research summaries for docs and frontend."""
import argparse
import json
import pickle
import shutil
from pathlib import Path
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--latency',type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    config=json.loads((args.benchmark/'benchmark.json').read_text())
    if config['dataset_mode']!='synthetic':
        p.error('This public demo export accepts synthetic results only.')
    for summary_path in args.benchmark.glob('family-*/summary.json'):
        summary=json.loads(summary_path.read_text())
        summary['dataset']='SYNTHETIC DEMONSTRATION (fixed dataset across seeds)'
        summary['configuration']['dataset_mode']='synthetic'
        summary_path.write_text(json.dumps(summary,indent=2))
        bundle_path=summary_path.parent/'pipeline.pkl'
        with bundle_path.open('rb') as stream:
            bundle=pickle.load(stream)
        bundle['demo']=True
        with bundle_path.open('wb') as stream:
            pickle.dump(bundle,stream)
    detail=pd.read_csv(args.benchmark/'detectability.csv')
    metrics=pd.read_csv(args.benchmark/'metrics.csv')
    expected=len(config['families'])*len(config['seeds'])
    frozen=detail[detail.model=='DQN frozen']
    if len(frozen)!=expected or frozen[['family','seed']].duplicated().any():
        p.error('Benchmark is incomplete or has duplicate folds.')
    destination=root/'docs/research-results';destination.mkdir(exist_ok=True)
    for name in ('benchmark.json','metrics.csv','summary.csv','detectability.csv','detectability.png'):
        shutil.copy2(args.benchmark/name,destination/name)
    latency=json.loads(args.latency.read_text())
    (destination/'latency.json').write_text(json.dumps(latency,indent=2))
    summaries=[]
    for family,rows in frozen.groupby('family'):
        rf=metrics[(metrics.family==family)&(metrics.model=='Random Forest')]
        summaries.append({'family':family,'seeds':len(rows),'recall_mean':float(rows.heldout_recall.mean()),
                          'recall_std':float(rows.heldout_recall.std(ddof=1)),
                          'distance_mean':float(rows.distance_mean.mean()),
                          'rf_recall_mean':float(rf.unseen_recall.mean())})
    report={'mode':'Synthetic demonstration','epochs':config['epochs'],'episodes':config['episodes'],
            'folds':expected,'families':summaries,'points':frozen.to_dict(orient='records'),'latency':latency}
    (root/'web/src/research-results.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    latex=[f"The fixed synthetic dataset was evaluated across {len(config['families'])} held-out families and {len(config['seeds'])} seeds ({expected} runs), with up to {config['epochs']} autoencoder epochs and {config['episodes']} DQN episodes per run. Table~\\ref{{tab:recall}} reports means and sample standard deviations, not confidence intervals.",
           '\\begin{table}[ht]\\centering\\caption{Synthetic held-out recall (five seeds).}\\label{tab:recall}',
           '\\begin{tabular}{lcc}\\toprule Family & Frozen DQN & RF mean \\\\ \\midrule']
    for row in summaries:
        latex.append(f"{row['family']} & {row['recall_mean']:.3f} $\\pm$ {row['recall_std']:.3f} & {row['rf_recall_mean']:.3f} \\\\")
    latex.extend(['\\bottomrule\\end{tabular}\\end{table}',
        '\\begin{figure}[ht]\\centering\\includegraphics[width=\\columnwidth]{detectability.png}\\caption{Frozen DQN detectability on synthetic data. Each point is one held-out-family/seed run.}\\end{figure}',
        'These results characterize this synthetic generator only. Cross-family distance comparisons are exploratory because each encoder is trained separately.'])
    (root/'paper/results.tex').write_text('\n'.join(latex))
    shutil.copy2(args.benchmark/'detectability.png',root/'paper/detectability.png')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':
    main()
