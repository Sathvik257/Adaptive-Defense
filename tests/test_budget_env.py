import numpy as np
from gymnasium.utils.env_checker import check_env
from adaptive_defense.budget_env import BudgetedDefenseEnv


def test_contract_and_budget():
    env=BudgetedDefenseEnv(np.ones((3,4)), [0,1,1],budget=1)
    check_env(env,skip_render_check=True)
    env.reset(seed=42)
    _,reward,_,_,info=env.step(2)
    assert reward==-.3 and info['analyst_label']==0 and info['remaining']==0
    _,reward,_,_,info=env.step(2)
    assert reward==-.3 and info['budget_exhausted'] and 'analyst_label' not in info
    _,reward,done,_,_=env.step(0)
    assert done and reward==-10


def test_labels_do_not_change_observations():
    a=BudgetedDefenseEnv(np.ones((2,3)),[0,0])
    b=BudgetedDefenseEnv(np.ones((2,3)),[1,1])
    np.testing.assert_array_equal(a.reset()[0],b.reset()[0])
    np.testing.assert_array_equal(a.step(0)[0],b.step(0)[0])
