"""Faithful reconstruction-residual attribution, not causal attack explanation."""
import numpy as np
import torch


def residual_explanations(model,features,names,top_k=3):
    x=np.asarray(features,dtype=np.float32)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all() or len(names)!=x.shape[1]:
        raise ValueError('Require finite features and aligned transformed feature names.')
    if not isinstance(top_k,int) or not 1<=top_k<=x.shape[1]:
        raise ValueError('top_k must be between one and feature count.')
    rows=[]
    with torch.no_grad():
        for batch in torch.as_tensor(x).split(1024):
            squared=(batch-model(batch)).square().numpy().astype(float)
            for values in squared:
                total=float(values.sum())
                order=np.argsort(-values,kind='stable')[:top_k]
                rows.append({'squared_error':total,'top_features':[
                    {'feature':str(names[i]),'squared_residual':float(values[i]),
                     'fraction_of_squared_error':float(values[i]/total) if total else 0.}
                    for i in order]})
    return rows
