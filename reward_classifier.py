from typing import Tuple, Dict
import random

import torch
import d4rl 
import gym

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F
# import pytorch_lightning as pl
# from pytorch_lightning.loggers import WandbLogger
import wandb


class RNNRewardPredictor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, lr, pos_weight=1, spread_len=0):
        super().__init__()
        self.lr = lr
        self.rnn = nn.LSTM(input_size=obs_dim+act_dim, hidden_size=hidden_dim, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim, 1)) # the sigmoid layer is in the loss
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        # self.loss = nn.BCELoss()
        self.spread_len = spread_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x.shape: (batch, seq, obs_dim+act_dim)
        rnn_output, _ = self.rnn(x)
        pred_reward = self.mlp(rnn_output)
        return pred_reward

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        seq, rewards = batch
        # delta_r = rewards[:, 1:] - rewards[:, :-1]

        # delta_r_pred = self.forward(seq[:, :-1])
        # loss = self.loss(delta_r, delta_r_pred)
        # return loss
        # if use spread reward
        r_pred = self.forward(seq[:, :-1])
        loss = self.loss(r_pred, rewards[:, 1:])
        return loss

    def predict(self, seq):
        seq = torch.as_tensor(seq).cuda().float().unsqueeze(0)
        with torch.no_grad():
            return self.forward(seq).squeeze().detach().cpu().numpy()

    def validation_step(self, batch, batch_idx):
        seq, rewards = batch

        # delta_r_pred = self.forward(seq[:, :-1])
        # delta_r_pred = (torch.sigmoid(delta_r_pred) > 0.5).int()
        # r0 = rewards[:, 0].unsqueeze(-1).repeat(1, delta_r_pred.shape[1], 1)
        # ones = torch.ones(delta_r_pred.shape).cuda()
        # r_pred = ones * r0
        # for t in range(delta_r_pred.shape[1]):
        #     mask = ones
        #     mask[:, :t] = 0
        #     r_pred = r_pred + torch.roll(delta_r_pred, t, 1) * mask

        # # loss = torch.sum(torch.abs(rewards[:, 1:] - r_pred)).item() 
        # loss = torch.sum(torch.abs(rewards[:, 1:] - rewards[:, 1:]*r_pred)).item()# this loss equals false negative
        # return loss, delta_r_pred

        delta_r_pred = self.forward(seq[:, :-1])
        # delta_r_pred = (delta_r_pred > 0.5).int()
        delta_r_pred = (torch.sigmoid(delta_r_pred) > 0.5).int()

        loss = torch.sum(torch.abs(rewards[:, 1:] - delta_r_pred)).item()# this loss equals false negative
        return loss, delta_r_pred

    # def predict(self, seq, r0=0):
    #     # seq: (seq_len, obs_dim + act_dim)
    #     seq = seq.unsqueeze(0)
    #     delta_r_pred = self.forward(seq)
    #     delta_r_pred = delta_r_pred.squeeze()
    #     delta_r_pred = (F.sigmoid(delta_r_pred) > 0.5).int()
    #     delta_r_pred[0] = 0
    #     ones = torch.ones(delta_r_pred.shape).cuda()
    #     r_pred = ones * r0
    #     for t in range(delta_r_pred.shape[0]):
    #         mask = ones
    #         mask[:t] = 0
    #         r_pred = r_pred + torch.roll(delta_r_pred, t, 0) * mask
    #     return r_pred

## some helper function that should be put into utils.py

class MLPSeqRewardPredictor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, lr, pos_weight=1, spread_len=0, seq_len=50):
        super().__init__()
        self.lr = lr
        self.mlp = nn.Sequential(nn.Linear((seq_len-2)*(obs_dim+act_dim+4), hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1)) # the sigmoid layer is in the loss
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.spread_len = spread_len

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x.shape: (batch, seq, obs_dim+act_dim+4)
        x = x.reshape(x.shape[0], -1)
        pred_reward = self.mlp(x)
        return pred_reward

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        seq, rewards = batch
        # delta_r = rewards[:, 1:] - rewards[:, :-1]

        r_pred = self.forward(seq[:, :-1])
        loss = self.loss(r_pred, rewards[:, -1])
        return loss

    def validation_step(self, batch, batch_idx):
        seq, rewards = batch

        delta_r_pred = self.forward(seq[:, :-1])
        delta_r_pred = (torch.sigmoid(delta_r_pred) > 0.5).int()

        loss = torch.sum(torch.abs(rewards[:, -1] - rewards[:, -1]*delta_r_pred)).item()# this loss equals false negative
        return loss, delta_r_pred

class MLPRewardPredictor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim, lr, batch_size=64, pos_weight=1, device='cuda'):
        super().__init__()
        self.lr = lr
        self.batch_size = batch_size
        print("mlp", 2*obs_dim+act_dim)
        self.mlp = nn.Sequential(nn.Linear(2*obs_dim+act_dim, hidden_dim), 
                                 nn.ReLU(), 
                                 nn.Linear(hidden_dim, hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_dim, 1)) # the sigmoid layer is in the loss
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.device = device
        self.opt = self.configure_optimizers()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x.shape: (batch, 2*obs_dim+act_dim)
        pred_reward = self.mlp(x)
        return pred_reward

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        sas, rewards = batch
        # delta_r = rewards[:, 1:] - rewards[:, :-1]
        r_pred = self.forward(sas)
        loss = self.loss(r_pred, rewards)
        return loss

    def validation_step(self, batch, batch_idx):
        seq, rewards = batch

        r_pred = self.forward(seq)
        r_pred = (torch.sigmoid(r_pred) > 0.5).int()

        loss = torch.sum(torch.abs(rewards - r_pred)).item() # only false negative
        return loss, r_pred

    def predict(self, obs, action, next_obs):
        obs = torch.as_tensor(obs).float()
        action = torch.as_tensor(action).float()
        next_obs = torch.as_tensor(next_obs).float()
        if len(obs.shape) < 2:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
            next_obs = action.unsqueeze(0)

        x = torch.cat((obs, action, next_obs), 1).to(self.device)

        with torch.no_grad():
            if len(obs.shape) < 2:
                return np.float_((torch.sigmoid(self(x)) > 0.5).float().item())
            else:
                return np.float_((torch.sigmoid(self(x)) > 0.5).float().squeeze().detach().cpu())


class RDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], seq_len=50, spread_len=5, norm=False, device='cuda'):
        # e.g. data = {'obs': np.array(1000, 8), 'act': np.array(1000, 12)} is flattened
        # seq = seq[60:] # only skill and ending state 
        #seq = seq[70:] # only ending state
        #seq = torch.cat((seq[:60], seq[70:]), 1) # only start and ending state
        self.sas = np.concatenate((data['observation'], data['action'], data['observation_next']), axis=1)
        # self.sas = data['observation_next']
        # self.sas = np.concatenate((data['observation'], data['observation_next']), axis=1)
        self.sas = torch.tensor(self.sas).float().to(device)
        if norm:
            self.sas = self.normalize(self.sas)
        self.reward = torch.tensor(data['reward']).float().to(device).unsqueeze(-1)
        self.reward = self.reward.reshape(self.reward.shape[0], 1)
        self.device = device
        self._len = len(data['reward'])

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # start index could be with in the range of [self.cut[i], self.cut[i+1]-50]
        return self.sas[index], self.reward[index]
    
    def normalize(self, x: torch.Tensor):
        return (x - x.mean(dim=1, keepdim=True))/x.std(dim=1, keepdim=True)

class RSeqDataset(Dataset):
    def __init__(self, data: Dict[str, np.ndarray], seq_len=50, spread_len=5, norm=True, device='cuda'):
        # e.g. data = {'obs': np.array(1000, 8), 'act': np.array(1000, 12)} is flattened
        self.seq = np.concatenate((data['observation'], data['action']), axis=1)
        self.seq = torch.tensor(self.seq).float().to(device)
        if norm:
            self.seq = self.normalize(self.seq)
        self.reward = torch.tensor(data['reward']).float().to(device).unsqueeze(-1)
        self.cut = [0] 
        for i in range(len(self.seq)):
            if data['done'][i]:
                self.cut.append(i)
        self._len = len(self.cut) - 1
        self.seq_len = seq_len
        self.spread_len = spread_len
        self.device = device

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        # start index could be with in the range of [self.cut[i], self.cut[i+1]-50]
        idx = random.randint(self.cut[index], self.cut[index+1]-self.seq_len)

        seq =  self.seq[idx : idx+self.seq_len]
        rewards = self.reward[idx : idx+self.seq_len] # rewards: (seq, 1)

        # task_onehot = torch.zeros((rewards.shape[0], 4)).to(self.device) # rewards: (seq, 1)
        # task_onehot.scatter_(1, rewards.long(), 1) # onehot: (seq, 4)
        # seq = torch.cat((seq, task_onehot), dim=-1) # use onehot to specify which task it currently is on


        # probability of event occur        
        # event_occur = rewards[1:] - rewards[:-1]
        # prob_rewards = event_occur
        # norm = np.exp(self.spread_len)
        # for t in range(1, self.spread_len+1):
        #     offset = torch.roll(event_occur, -t, 0)
        #     offset[-t:] = 0.0
        #     prob_rewards = prob_rewards + offset * np.exp(self.spread_len-t) / norm

        event_occur = ((rewards[1:] - rewards[:-1]) > 0).float()

        # repeat event occur, keep as binary label
        # spread_rewards = torch.zeros(event_occur.shape).to(self.device)
        # for t in range(-self.spread_len, self.spread_len):
        #     offset = torch.roll(event_occur, t, 0)
        #     if t < 0:
        #         offset[t:] = 0.0
        #     else:
        #         offset[:t] = 0.0
        #     spread_rewards = spread_rewards + offset
        # if self.spread_len == 0:
        #     spread_rewards = event_occur
        # return seq[1:], spread_rewards

        return seq[1:], event_occur
    
    def normalize(self, x: torch.Tensor):
        return (x - x.mean(dim=1, keepdim=True))/x.std(dim=1, keepdim=True)
