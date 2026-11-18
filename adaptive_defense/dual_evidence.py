"""Dual evidence from frozen AE reconstruction and benign latent geometry."""
import numpy as np
import torch
from sklearn.covariance import LedoitWolf


def encode(model, features):
    x = np.asarray(features,dtype=np.float32)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('Need a nonempty finite feature matrix.')
    with torch.no_grad():
        return np.concatenate([model.net[:5](b).numpy() for b in torch.as_tensor(x).split(1024)])


class DualEvidence:
    """Fit geometry and empirical score scaling on benign training data only.

    Max fusion is fixed before evaluation; no supervised weight selection.
    Outputs are empirical ranks, not calibrated attack probabilities.
    """
    def fit(self, reconstruction, latent):
        reconstruction = np.asarray(reconstruction,dtype=float)
        latent = np.asarray(latent,dtype=float)
        if latent.ndim != 2 or len(latent)<2 or reconstruction.shape != (len(latent),) or not np.isfinite(latent).all() or not np.isfinite(reconstruction).all():
            raise ValueError('Need aligned finite training evidence and >=2 records.')
        self.geometry = LedoitWolf().fit(latent)
        self.reference = [np.sort(reconstruction),np.sort(self.geometry.mahalanobis(latent))]
        return self

    def transform(self, reconstruction, latent):
        reconstruction = np.asarray(reconstruction,dtype=float)
        latent = np.asarray(latent,dtype=float)
        if latent.ndim != 2 or reconstruction.shape != (len(latent),) or not len(latent) or not np.isfinite(latent).all() or not np.isfinite(reconstruction).all():
            raise ValueError('Need aligned finite evaluation evidence.')
        distance = self.geometry.mahalanobis(latent)
        ranks = np.column_stack([np.searchsorted(ref,values,side='right')/(len(ref)+1)
                                 for ref,values in zip(self.reference,(reconstruction,distance))])
        return {'reconstruction':reconstruction,'latent_distance':distance,
                'dual_max':ranks.max(axis=1),'rank_pairs':ranks}
