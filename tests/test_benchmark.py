import numpy as np
import pandas as pd
import pytest
from adaptive_defense.benchmark import centroid_distance
from adaptive_defense.data import split_data


def test_distance_uses_nearest_class_not_pooled_mean():
    training=np.array([[0.,0.],[2.,2.],[10.,10.],[12.,12.]])
    query=np.array([[1.,1.],[11.,11.],[6.,6.]])
    distance=centroid_distance(training,['a','a','b','b'],query)
    np.testing.assert_allclose(distance[:2],0)
    assert distance[2]>0


def test_distance_invariant_to_coordinate_units():
    x=np.array([[0.,1.],[2.,4.],[8.,8.],[10.,12.]])
    q=np.array([[6.,6.]])
    np.testing.assert_allclose(centroid_distance(x,['a','a','b','b'],q),
                              centroid_distance(x*[100,2],['a','a','b','b'],q*[100,2]))


def test_duplicate_family_conflicts_rejected():
    frame=pd.DataFrame({'value':[1,1],'label':[1,1],'attack_cat':['scan','exploit']})
    with pytest.raises(ValueError,match='conflicting attack families'):
        split_data(frame,['scan'],42)
