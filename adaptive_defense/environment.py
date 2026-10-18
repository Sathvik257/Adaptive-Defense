"""Threshold-control environment with no access to current labels in observations."""
import numpy as np

class ThresholdEnvironment:
    def __init__(self,scores,labels,grid,window=256,feedback=False,fn_cost=5):
        self.scores=np.asarray(scores)
        self.labels=np.asarray(labels)
        self.grid=np.asarray(grid)
        self.window=window
        self.feedback=feedback
        self.fn_cost=fn_cost
        self.scale=max(float(grid[-1]),1e-8)
        self.reset()

    def reset(self):
        self.position=0
        self.index=len(self.grid)//2
        self.state=np.array([self.index/(len(self.grid)-1),0,0,0,0,0,0,0],dtype=np.float32)
        return self.state.copy()

    def step(self,action):
        if self.position>=len(self.scores):
            raise RuntimeError("Episode already complete.")
        if action not in (0,1,2):
            raise ValueError("Action must be 0, 1 or 2.")
        self.index=int(np.clip(self.index+action-1,0,len(self.grid)-1))
        end=min(self.position+self.window,len(self.scores))
        scores=self.scores[self.position:end]
        labels=self.labels[self.position:end]
        pred=(scores>self.grid[self.index]).astype(int)
        fn=int(((pred==0)&(labels==1)).sum())
        fp=int(((pred==1)&(labels==0)).sum())
        reward=-(self.fn_cost*fn+fp)/len(scores)
        self.state=np.array([self.index/(len(self.grid)-1),
            np.clip(scores.mean()/self.scale,0,20),np.clip(np.quantile(scores,.95)/self.scale,0,20),
            pred.mean(),action-1,fn/len(scores) if self.feedback else 0,
            fp/len(scores) if self.feedback else 0,float(self.feedback)],dtype=np.float32)
        info={"start":self.position,"end":end,"threshold":float(self.grid[self.index]),
              "alerts":int(pred.sum()),"reward":reward,"predictions":pred}
        self.position=end
        return self.state.copy(),reward,end==len(self.scores),info

def rollout(agent,scores,labels,grid,window=256,train=False,epsilon=0,feedback=False,rule=False):
    env=ThresholdEnvironment(scores,labels,grid,window,feedback)
    state=env.reset()
    records=[]; predictions=[]; done=False
    while not done:
        if rule:
            action=2 if state[3]>.10 else (0 if state[3]<.02 and env.position else 1)
        else:
            action=agent.action(state,epsilon)
        nxt,reward,done,info=env.step(action)
        if train:
            agent.learn((state,action,reward,nxt,done))
        predictions.extend(info.pop("predictions").tolist())
        info["action"]=action
        records.append(info)
        state=nxt
    return np.array(predictions),records
