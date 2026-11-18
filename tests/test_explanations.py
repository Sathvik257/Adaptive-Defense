import numpy as np
import pytest
import torch
from adaptive_defense.explanations import residual_explanations

class Zero(torch.nn.Module):
    def forward(self,x):return torch.zeros_like(x)


def test_contributions_reconcile_and_rank():
    row=residual_explanations(Zero(),[[1,2,3]],['a','b','c'])[0]
    assert row['squared_error']==14
    assert [r['feature'] for r in row['top_features']]==['c','b','a']
    assert sum(r['fraction_of_squared_error'] for r in row['top_features'])==pytest.approx(1)


def test_zero_error_and_invalid_names():
    row=residual_explanations(torch.nn.Identity(),[[1,2]],['a','b'],2)[0]
    assert row['squared_error']==0
    assert all(r['fraction_of_squared_error']==0 for r in row['top_features'])
    with pytest.raises(ValueError):residual_explanations(Zero(),[[1,2]],['a'])
