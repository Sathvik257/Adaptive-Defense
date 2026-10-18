"""Score anonymous SUS responses, without generating or imputing participant data."""
import argparse
import json
import numpy as np
import pandas as pd

def score_responses(frame):
    columns=[f'q{i}' for i in range(1,11)]
    values=frame[columns].apply(pd.to_numeric,errors='raise').to_numpy()
    if not len(values) or not np.isfinite(values).all() or not np.isin(values,[1,2,3,4,5]).all():
        raise ValueError('Each participant must answer all ten items with integers 1–5.')
    contributions=np.where(np.arange(10)%2==0,values-1,5-values)
    return contributions.sum(axis=1)*2.5

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    args=parser.parse_args()
    scores=score_responses(pd.read_csv(args.csv))
    print(json.dumps({'participants':len(scores),'scores':scores.tolist(),'mean':float(scores.mean()),
                      'sample_std':float(scores.std(ddof=1)) if len(scores)>1 else None,
                      'note':'SUS is a 0–100 scale, not a percentage; small samples are uncertain.'},indent=2))
