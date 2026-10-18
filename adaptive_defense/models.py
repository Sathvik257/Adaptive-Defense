"""Compact PyTorch models and training helpers."""
import random
from collections import deque
import numpy as np
import torch
from torch import nn

class Autoencoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),
                                 nn.Linear(64,16),nn.Linear(16,64),nn.ReLU(),nn.Linear(64,128),
                                 nn.ReLU(),nn.Linear(128,dimension))
    def forward(self,x):
        return self.net(x)

def train_autoencoder(train, validation, epochs, seed):
    torch.manual_seed(seed)
    model = Autoencoder(train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    x = torch.tensor(train,dtype=torch.float32)
    v = torch.tensor(validation,dtype=torch.float32)
    history, best, patience, checkpoint = [], float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        losses=[]
        for idx in torch.randperm(len(x)).split(256):
            loss = nn.functional.mse_loss(model(x[idx]),x[idx])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(v),v).item()
        history.append({"epoch":epoch+1,"training_loss":float(np.mean(losses)),"validation_loss":val})
        if val < best:
            best, patience = val, 0
            checkpoint = {k:t.detach().clone() for k,t in model.state_dict().items()}
        else:
            patience += 1
        if patience >= 8:
            break
    model.load_state_dict(checkpoint)
    return model, history

def anomaly_scores(model, x):
    model.eval()
    with torch.no_grad():
        return np.concatenate([((model(t)-t)**2).mean(1).numpy()
            for t in torch.tensor(x,dtype=torch.float32).split(1024)])

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(8,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,3))
    def forward(self,x):
        return self.net(x)

class DQN:
    def __init__(self,seed=42):
        torch.manual_seed(seed)
        self.rng=random.Random(seed)
        self.online,self.target=QNetwork(),QNetwork()
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer=torch.optim.Adam(self.online.parameters(),lr=3e-4)
        self.memory=deque(maxlen=20000)
        self.updates=0

    def action(self,state,epsilon=0):
        if self.rng.random()<epsilon:
            return self.rng.randrange(3)
        with torch.no_grad():
            return int(self.online(torch.tensor(state,dtype=torch.float32)).argmax().item())

    def learn(self,transition):
        self.memory.append(transition)
        if len(self.memory)<64:
            return None
        batch=self.rng.sample(list(self.memory),64)
        s,a,r,n,d=zip(*batch)
        states=torch.tensor(np.array(s),dtype=torch.float32)
        nxt=torch.tensor(np.array(n),dtype=torch.float32)
        q=self.online(states).gather(1,torch.tensor(a).unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target=torch.tensor(r,dtype=torch.float32)+.95*(1-torch.tensor(d,dtype=torch.float32))*self.target(nxt).max(1).values
        loss=nn.functional.smooth_l1_loss(q,target)
        self.optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(),1.0)
        self.optimizer.step(); self.updates+=1
        if self.updates%500==0:
            self.target.load_state_dict(self.online.state_dict())
        return loss.item()
