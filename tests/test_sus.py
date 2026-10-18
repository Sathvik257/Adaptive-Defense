import pandas as pd
import pytest
from tools.score_sus import score_responses

def test_sus_extremes_and_validation():
    columns=[f'q{i}' for i in range(1,11)]
    frame=pd.DataFrame([[5,1]*5,[1,5]*5,[3]*10],columns=columns)
    assert score_responses(frame).tolist()==[100,0,50]
    frame.loc[0,'q1']=6
    with pytest.raises(ValueError):
        score_responses(frame)
