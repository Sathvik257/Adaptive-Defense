"""Train PPO on a labeled NPZ training split and evaluate a separate test split.

NPZ requires x_train, y_train, x_test, y_test. Features must be prepared using
training-only fitting; never supply labels or identifiers as feature columns.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from adaptive_defense.budget_env import BudgetedDefenseEnv

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.output.exists() or args.steps<64:
        parser.error('Choose a new output directory and at least 64 steps.')
    data=np.load(args.data,allow_pickle=False)
    train=BudgetedDefenseEnv(data['x_train'],data['y_train'])
    test=BudgetedDefenseEnv(data['x_test'],data['y_test'])
    if train.observation_space.shape!=test.observation_space.shape:
        parser.error('Training and test feature dimensions must match.')
    model=PPO('MlpPolicy',train,seed=args.seed,n_steps=64,batch_size=64,device='cpu',verbose=0)
    model.learn(total_timesteps=args.steps)
    state,_=test.reset(seed=args.seed)
    done=False; costs=[]; queries=0
    while not done:
        action,_=model.predict(state,deterministic=True)
        state,_,done,_,info=test.step(int(action))
        costs.append(info['cost']);queries+=int(info['queried'])
    args.output.mkdir(parents=True)
    model.save(args.output/'ppo')
    report={'seed':args.seed,'steps':model.num_timesteps,'cost':sum(costs),'queries':queries,
            'cumulative_cost':np.cumsum(costs).tolist(),
            'assumption':'Perfect immediate analyst; exhausted query falls back to alert.'}
    (args.output/'evaluation.json').write_text(json.dumps(report,indent=2))
    print(f'Evaluation saved to {args.output}')

if __name__=='__main__':
    main()
