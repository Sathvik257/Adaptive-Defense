"""Architecture facts from the author's later public code, separate from paper facts."""
from torch import nn

AUTHOR_REVISION = '4b291b1b9c05c748f61882921a1aab86101f0954'
AUTHOR_URL = f'https://github.com/XueYang0130/ADT/tree/{AUTHOR_REVISION}'
PAPER_SETTINGS = {
    'Yahoo': {'episodes':20000, 'tau':10, 'k':2, 'l':1, 'alpha':.9, 'beta':.1},
    'SWaT': {'episodes':20000, 'tau':12, 'k':2, 'l':10, 'alpha':.9, 'beta':.1},
    'WADI': {'episodes':20000, 'tau':10, 'k':2, 'l':10, 'alpha':.9, 'beta':.1},
}


class ReferenceAutoencoder(nn.Module):
    """Independent PyTorch implementation of the published reference topology.

    The 2023 PDF does not specify widths. Later author code uses a linear latent
    layer and sigmoid reconstruction. Latent 100 is its multivariate setting;
    using it on UNSW (and using tau=1) remains an explicit project adaptation.
    """
    def __init__(self, dimension, latent=100):
        super().__init__()
        if dimension < 4 or latent < 1:
            raise ValueError('Reference topology requires input dimension >=4 and positive latent size.')
        self.net = nn.Sequential(
            nn.Linear(dimension,dimension//2), nn.ReLU(),
            nn.Linear(dimension//2,dimension//4), nn.ReLU(),
            nn.Linear(dimension//4,latent),
            nn.Linear(latent,dimension//4), nn.ReLU(),
            nn.Linear(dimension//4,dimension//2), nn.ReLU(),
            nn.Linear(dimension//2,dimension), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)


def architecture_manifest(dimension, latent=100):
    return {'source':AUTHOR_URL, 'source_scope':'later author code; not proof of the 2023 experimental topology',
            'implementation':'independent PyTorch implementation; author code uses TensorFlow/Keras',
            'ae_widths':[dimension,dimension//2,dimension//4,latent,dimension//4,dimension//2,dimension],
            'ae_activations':['ReLU','ReLU','linear','ReLU','ReLU','sigmoid'],
            'q_widths':[6,32,32,16,2], 'q_activations':['ReLU','ReLU','ReLU','linear'],
            'state_order':['mean','sample_variance','TP/k','TN/k','FP/k','FN/k'],
            'score':'L2 / frozen train maximum, clipped to [0,1]',
            'ae_loss':'mean per-window L2 (2023 Equation 4, not later code MSE)',
            'reward':'alpha*(TP-FP-FN)+(1-alpha)*TN; updated rolling k history',
            'comparison':'strict score > threshold', 'paper_settings':PAPER_SETTINGS}
