import numpy as np
import pytest
import torch
from adaptive_defense.dual_evidence import DualEvidence, encode
from adaptive_defense.adt_architecture import ReferenceAutoencoder


def test_latent_signal_and_frozen_reference():
    rng=np.random.default_rng(4)
    z=rng.normal(size=(100,3)); r=rng.uniform(size=100)
    detector=DualEvidence().fit(r,z)
    before=[v.copy() for v in detector.reference]
    out=detector.transform([.2,.2],[[0,0,0],[20,20,20]])
    assert out['latent_distance'][1]>out['latent_distance'][0]
    assert out['dual_max'][1]>out['dual_max'][0]
    assert np.all((out['rank_pairs']>=0)&(out['rank_pairs']<=1))
    for a,b in zip(before,detector.reference):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError):
        detector.transform([np.nan],[[0,0,0]])


def test_encoder_boundary_matches_linear_latent():
    model=ReferenceAutoencoder(8,latent=5).eval()
    x=np.ones((3,8),dtype=np.float32)
    expected=model.net[:5](torch.tensor(x)).detach().numpy()
    np.testing.assert_allclose(encode(model,x),expected)
    assert expected.shape==(3,5)
