"""Section 5.1 metrics and ten-block robustness analysis from the supplied paper."""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paper_metrics(labels, predictions):
    y, p = np.asarray(labels), np.asarray(predictions)
    if y.ndim != 1 or not len(y) or y.shape != p.shape or not np.isin(y,[0,1]).all() or not np.isin(p,[0,1]).all():
        raise ValueError('Metrics require aligned nonempty binary arrays.')
    tp = int(((y==1)&(p==1)).sum())
    fp = int(((y==0)&(p==1)).sum())
    fn = int(((y==1)&(p==0)).sum())
    if tp == fp == fn == 0:
        precision = recall = f1 = 1.
    elif tp == 0:
        precision = recall = f1 = 0.
    else:
        precision, recall = tp/(tp+fp), tp/(tp+fn)
        f1 = 2*precision*recall/(precision+recall)
    return {'precision':precision,'recall':recall,'f1':f1}


def robustness(labels, predictions, baseline, blocks=10):
    """Even contiguous blocks in supplied replay order, not invented chronology."""
    if blocks < 2 or len(labels) < blocks or baseline not in predictions:
        raise ValueError('Need >=2 nonempty blocks and an included baseline.')
    for values in predictions.values():
        paper_metrics(labels,values)
    rows = []
    for block, indices in enumerate(np.array_split(np.arange(len(labels)),blocks)):
        for name,pred in predictions.items():
            rows.append({'model':name,'block':block,'rows':len(indices),
                         **paper_metrics(np.asarray(labels)[indices],np.asarray(pred)[indices])})
    frame = pd.DataFrame(rows)
    comparison = []
    reference = frame[frame.model==baseline].f1.to_numpy()
    for name in predictions:
        subset = frame[frame.model==name]
        delta = subset.f1.to_numpy()-reference
        pvalue = 1. if np.all(delta==0) else float(wilcoxon(delta,alternative='two-sided').pvalue)
        comparison.append({'model':name,'baseline':baseline,
                           'f1_mean':float(subset.f1.mean()),'f1_sample_sd':float(subset.f1.std(ddof=1)),
                           'wilcoxon_two_sided_p':pvalue,'p_below_0_05':pvalue<.05})
    return frame,pd.DataFrame(comparison)
