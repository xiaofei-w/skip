import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
from scipy.stats import norm
import itertools
import tqdm
import copy
import scipy.stats as st
import os

from kitchen_info import *

device = 'cuda'
# device = 'cpu'

def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

def KMeans(x, K=3, Niter=50, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    N, D = x.shape  # Number of samples, dimension of the ambient space

    c = x[:K, :].clone()    # Simplistic initialization for the centroids

    x_i = x.view(N, 1, D)  # (N, 1, D) samples
    c_j = c.view(1, K, D)  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:,None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average
        
    center_index = D_ij.argmin(dim=0).long().view(-1)
    center = x[center_index]
    
    return  center_index

class RewardModel:
    def __init__(self, ds, da, 
                 ensemble_size=3, lr=3e-4, mb_size = 128, size_segment=1, 
                 env_maker=None, max_size=100, activation='tanh', capacity=5e5,  
                 teacher_type=0, teacher_noise=0.0, teacher_margin=0.0, teacher_thres=0.0, 
                 large_batch=1, bc_capa=1, task_seq='mkb'):
        # train data is trajectories, must process to sa and s..   
        self.ds = ds
        self.da = da
        self.de = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.opt = None
        self.model = None
        self.max_size = max_size # maximum # of episodes for training
        self.activation = activation
        self.size_segment = size_segment
        self.current_mode = 1
        self.max_samples = 10000
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.ds+self.da), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
                
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.raw_actions = []
        self.img_inputs = []
        self.mb_size = mb_size
        self.train_batch_size = 128
        self.MSEloss = nn.MSELoss()
        self.CEloss = nn.CrossEntropyLoss()
        self.running_means = []
        self.running_stds = []
        self.best_seg = []
        self.best_label = []
        self.best_action = []
        self.teacher_type = teacher_type
        self.teacher_noise = teacher_noise
        self.teacher_margin = teacher_margin
        self.teacher_thres = teacher_thres
        self.large_batch = large_batch
        self.buffer_capa = int(mb_size*0.5*bc_capa)
        self.bc_buffer_state = np.empty((self.buffer_capa, size_segment, self.ds), dtype=np.float32)
        self.bc_buffer_action = np.empty((self.buffer_capa, size_segment, self.da), dtype=np.float32)
        self.bc_buffer_index = 0
        self.bc_buffer_full = False
        self.bc_capa = bc_capa
        
        file_name = os.getcwd()+'/sampling_log.txt'
        self.f_io = open(file_name, 'a')
        self.round_counter = 0

        self.task_seq = [task_dict[k] for k in task_seq]
        
    def increase_mode(self):
        self.current_mode = self.current_mode + 1
        
    def construct_ensemble(self):
        for i in range(self.de):
            model = nn.Sequential(*gen_net(in_size=self.ds+self.da, out_size=1, H=256, n_layers=3, activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
            
        self.opt = torch.optim.Adam(self.paramlst, lr = self.lr)
            
    def add_data(self, obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)

        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
        elif done:
            # print(self.inputs[-1])
            # print(len(self.inputs[-1]))
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([])
            self.targets.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])

        

    def add_data_in_batch(self, obs, act, rew, done):
        # done[-1] = np.logical_not(done[-1]) # to finish the last traj no matter it is complete or not
        b = len(obs)
        for i in range(b):
            # print(obs[i].shape,act[i].shape)
            self.add_data(obs[i], act[i], rew[i], done[i])

        # print(done)

        # for i in range(len(self.inputs)):
        #     assert len(self.inputs[i]) <= 28

        # for i in range(len(self.inputs)):
        #     print(i, len(self.inputs[i]))
        # self.remove_incomplete_traj()
        # print()

    
    def remove_incomplete_traj(self, complete_len=28):
        print("before removing, length of self.inputs:", len(self.inputs))
        del_idx = []
        for i in range(len(self.inputs)-1):
            if len(self.inputs[i]) != complete_len:
                del_idx.insert(0, i)
                print("deleted:", i, self.inputs[i].shape)
        self.inputs[-1] = []
        for i in del_idx:
            del self.inputs[i]
            del self.targets[i]

        try:
            for i in range(len(self.inputs)-1):
                assert len(self.inputs[i]) == complete_len
        except:
            print([len(self.inputs[i]) for i in range(len(self.inputs))])

        print("after removing, length of self.inputs:", len(self.inputs))
                
    def add_data_with_rawact(self, obs, act, raw_act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)
        flat_act = raw_act.reshape(1, self.da)
        
        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            self.raw_actions.append(flat_act)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            self.raw_actions[-1] = np.concatenate([self.raw_actions[-1], flat_act])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.raw_actions = self.raw_actions[1:]
            self.inputs.append([])
            self.targets.append([])
            self.raw_actions.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
                self.raw_actions[-1] = flat_act
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                self.raw_actions[-1] = np.concatenate([self.raw_actions[-1], flat_act])
                
    def add_data_with_img(self, obs, img_obs, act, rew, done):
        sa_t = np.concatenate([obs, act], axis=-1)
        r_t = rew
        
        flat_input = sa_t.reshape(1, self.da+self.ds)
        r_t = np.array(r_t)
        flat_target = r_t.reshape(1, 1)
        
        img_obs = np.expand_dims(img_obs, axis=0)
        
        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            self.img_inputs.append(img_obs)
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], img_obs])
            # FIFO
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
                self.img_inputs = self.img_inputs[1:]
            self.inputs.append([])
            self.targets.append([])
            self.img_inputs.append([])
        else:
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
                self.img_inputs[-1] = img_obs
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
                self.img_inputs[-1] = np.concatenate([self.img_inputs[-1], img_obs])
                
    def add_data_batch(self, obses, rewards):
        num_env = obses.shape[0]
        for index in range(num_env):
            self.inputs.append(obses[index])
            self.targets.append(rewards[index])
    
    def save_data(self, save_dir):
        np.save(save_dir + '/inputs.npy', self.inputs)
        np.save(save_dir + '/targets.npy', self.targets)
        
    def load_data(self, save_dir, num_traj):
        self.inputs = np.load(save_dir + '/inputs.npy', allow_pickle=True)
        self.targets = np.load(save_dir + '/targets.npy', allow_pickle=True)
        self.inputs = self.inputs[:num_traj]
        self.targets = self.targets[:num_traj]
        
        loaded_inputs, loaded_targets = [], []
        for index_ in range(num_traj):
            loaded_inputs.append(self.inputs[index_])
            loaded_targets.append(self.targets[index_])
        self.inputs = np.array(loaded_inputs)
        self.targets = np.array(loaded_targets)
        
    def get_rank_probability(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        # get probability x_1 > x_2
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)

    def get_dominance(self, x_1, x_2, sigma=1.):
        probs = []
        for member in range(self.de):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        probs = np.sort(probs, axis=0)
        return probs[0] > 0.5

    def p_hat_member(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        # taking 0 index for probability x_1 > x_2
        return F.softmax(r_hat, dim=-1)[:,0]
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        # softmaxing to get the probabilities according to eqn 1
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member)
            r_hat2 = self.r_hat_member(x_2, member=member)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent

    def r_hat_member(self, x, member=-1):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.ensemble[member](torch.from_numpy(x).float().to(device))

    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        for member in range(self.de):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)

        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt' % (model_dir, step, member)
            )
            
    def save_last(self, model_dir):
        for member in range(self.de):
            torch.save(
                self.ensemble[member].state_dict(), '%s/reward_model_%s_last.pt' % (model_dir, member)
            )
            
    def load(self, model_dir, step):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_%s.pt' % (model_dir, step, member))
            )
            
    def load_last(self, model_dir):
        for member in range(self.de):
            self.ensemble[member].load_state_dict(
                torch.load('%s/reward_model_%s_last.pt' % (model_dir, member))
            )
    
    def get_train_acc(self):
        ensemble_acc = np.array([0 for _ in range(self.de)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len/batch_size))
        
        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch+1)*batch_size
            if (epoch+1)*batch_size > max_len:
                last_index = max_len
                
            sa_t_1 = self.buffer_seg1[epoch*batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch*batch_size:last_index]
            labels = self.buffer_label[epoch*batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            for member in range(self.de):
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
        ensemble_acc = ensemble_acc / total
        return np.mean(ensemble_acc)
    
    def get_queries(self, mb_size=20, img_flag=True, recent_flag=False):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1

        if len(self.inputs[-2]) < len_traj: # change # could be inputs[-1] is [], while inputs[-2] is 20? why?
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len]) # here, nested numpy array didnt turn into full np array. should be of shape(11, 70), but only(11,), because self.inputs has traj that are not of full length
        train_targets = np.array(self.targets[:max_len])

        # get last traj
        if recent_flag:
            if len(self.inputs[-1]) == len_traj:
                last_index = -1
            else:
                last_index = -2
            train_last_inputs = np.array(self.inputs[last_index])
            train_last_inputs = np.expand_dims(train_last_inputs, axis=0)        
            train_last_inputs = train_last_inputs.repeat(int(mb_size), axis=0)
            train_last_targets = np.array(self.targets[last_index])
            train_last_targets = np.expand_dims(train_last_targets, axis=0)
            train_last_targets = train_last_targets.repeat(int(mb_size), axis=0)
                
        if img_flag:
            train_img_inputs = np.array(self.img_inputs[:max_len])
            if recent_flag:
                train_last_img = np.array(self.img_inputs[last_index])
                train_last_img = np.expand_dims(train_last_img, axis=0)
                train_last_img = train_last_img.repeat(int(mb_size), axis=0)
                    
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t_2 = train_inputs[batch_index_2] # Batch x T x dim of s&a
        r_t_2 = train_targets[batch_index_2] # Batch x T x 1
        
        if recent_flag:
            total_sa_t = np.concatenate([sa_t_2, train_last_inputs], axis=0)
            total_r_t = np.concatenate([r_t_2, train_last_targets], axis=0)
            random_index = np.random.permutation(2*mb_size)
            
            sa_t_1 = total_sa_t[random_index[:mb_size]]
            sa_t_2 = total_sa_t[random_index[mb_size:]]
            r_t_1 = total_r_t[random_index[:mb_size]]
            r_t_2 = total_r_t[random_index[mb_size:]]
            if img_flag:
                img_t_2 = train_img_inputs[batch_index_2]
                total_img_t = np.concatenate([img_t_2, train_last_img], axis=0)
                img_t_1 = total_img_t[random_index[:mb_size]]
                img_t_2 = total_img_t[random_index[mb_size:]]
                img_t_1 = img_t_1.reshape(-1, *img_t_1.shape[-3:])
                img_t_2 = img_t_2.reshape(-1, *img_t_2.shape[-3:])
        else:
            batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
            sa_t_1 = train_inputs[batch_index_1] # Batch x T x dim of s&a
            r_t_1 = train_targets[batch_index_1] # Batch x T x 1

            if img_flag:
                img_t_1 = train_img_inputs[batch_index_1]
                img_t_2 = train_img_inputs[batch_index_2]
                img_t_1 = img_t_1.reshape(-1, *img_t_1.shape[-3:])
                img_t_2 = img_t_2.reshape(-1, *img_t_2.shape[-3:])
                
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (Batch x T) x dim of s&a
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (Batch x T) x 1
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (Batch x T) x dim of s&a
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (Batch x T) x 1

        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_2 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment, size=mb_size, replace=True).reshape(-1,1)
        

        # assert(sa_t_1.shape[-1]==70)
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
                
        if img_flag:
            img_t_1 = np.take(img_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
            img_t_2 = np.take(img_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
            return sa_t_1, sa_t_2, img_t_1, img_t_2
        else:
            return sa_t_1, sa_t_2, r_t_1, r_t_2
    
    def get_queries_one_seg(self, seg_margin=10, mb_size=20, img_flag=True, recent_flag=False):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        img_t_1, img_t_2 = None, None

        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        # get train traj
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
            
        if img_flag:
            train_img_inputs = np.array(self.img_inputs[:max_len])
            
        batch_index = np.random.choice(max_len, size=mb_size, replace=True)
        sa_t = train_inputs[batch_index] # Batch x T x dim of s&a
        r_t = train_targets[batch_index] # Batch x T x 1
        sa_t = sa_t.reshape(-1, sa_t.shape[-1]) # (Batch x T) x dim of s&a
        r_t = r_t.reshape(-1, r_t.shape[-1]) # (Batch x T) x 1

        if img_flag:
            img_t = train_img_inputs[batch_index]
            img_t = img_t.reshape(-1, *img_t.shape[-3:])
        
        # Generate time index 
        time_index = np.array([list(range(i*len_traj,
                                            i*len_traj+self.size_segment)) for i in range(mb_size)])
        time_index_1 = time_index + np.random.choice(len_traj-self.size_segment-seg_margin-1, size=mb_size, replace=True).reshape(-1,1)
        time_index_2 = time_index_1 + seg_margin

        sa_t_1 = np.take(sa_t, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t, time_index_2, axis=0) # Batch x size_seg x 1
        
        if img_flag:
            img_t_1 = np.take(img_t, time_index_1, axis=0) # Batch x size_seg x dim of s&a
            img_t_2 = np.take(img_t, time_index_2, axis=0) # Batch x size_seg x dim of s&a
            return sa_t_1, sa_t_2, img_t_1, img_t_2
        else:
            return sa_t_1, sa_t_2, r_t_1, r_t_2

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index

    def get_next_goal(self, s):
        prev_finished = -1
        for i, task in enumerate(self.task_seq):
            element_idx = OBS_ELEMENT_INDICES[task]
            goal = np.zeros_like(s) 
            goal[element_idx] = OBS_ELEMENT_GOALS[task]
            dist = np.linalg.norm(s[..., element_idx] - goal[element_idx])
            if dist < 0.3 and i == prev_finished + 1:
                prev_finished = i
        return prev_finished + 1

            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=0):
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        num_noise = 0
        
        if self.teacher_type == 0: # ideal teacher
            # labels = 1*(sum_r_t_1 <= sum_r_t_2) - 0.5*(abs(sum_r_t_1 - sum_r_t_2) < 0.1) # temp, for soft preference
            
            ## only  microwave ##
            # batch = sa_t_1.shape[0]
            # labels = np.zeros(sum_r_t_1.shape)
            # for i in range(batch):
            #     if sum_r_t_1[i] > 0 or sum_r_t_2[i] > 0: 
            #         labels[i] = 1*(sum_r_t_1[i] < sum_r_t_2[i]) # 0.5,0.5 by expectationz
            #     else:
            #         sa1_last = sa_t_1[i, -1]; sa2_last = sa_t_2[i, -1] 
            #         sa1_first = sa_t_1[i, -1]; sa2_first = sa_t_2[i, -1] 
            #         labels[i] = (sa1_first[22] - sa1_last[22]) < (sa2_first[22] - sa2_last[22])

            ## three tasks in order ###
            batch = sa_t_1.shape[0]
            labels = np.zeros(sum_r_t_1.shape)
            for i in range(batch):
                if sum_r_t_1[i] > 0 or sum_r_t_2[i] > 0: 
                    labels[i] = 1*(sum_r_t_1[i] < sum_r_t_2[i]) # 0.5,0.5 by expectationz
                else:
                    # only consider the case where for both skill the task has not completed
                    ###
                    task_list = self.task_seq #['microwave', 'kettle', 'bottom burner']
                    sa1_last = sa_t_1[i, -1]; sa2_last = sa_t_2[i, -1] 
                    sa1_first = sa_t_1[i, -1]; sa2_first = sa_t_2[i, -1] 
                    
                    goal1 = self.get_next_goal(sa1_last)
                    goal2 = self.get_next_goal(sa2_last)

                    # if goal1 == 3 or goal2 == 3:
                        # print("goal1/2", goal1, goal2)
                        # print("len task list", len(task_list))
                    if (goal1 != goal2) or (goal1 >= len(task_list)) or (goal2 >= len(task_list)):
                        labels[i] = (goal1 < goal2) # 1 if (progress2 > 0) else 0
                    else:
                        task = task_list[goal1]
                        element_idx = OBS_ELEMENT_INDICES[task]
                        goal = np.zeros_like(sa1_last) 
                        goal[element_idx] = OBS_ELEMENT_GOALS[task]
                        progress1 = np.linalg.norm(sa1_first[..., element_idx] - goal[element_idx]) - np.linalg.norm(sa1_last[..., element_idx] - goal[element_idx])           

                        task = task_list[goal2]
                        element_idx = OBS_ELEMENT_INDICES[task]
                        goal = np.zeros_like(sa1_last) 
                        goal[element_idx] = OBS_ELEMENT_GOALS[task]
                        progress2 = np.linalg.norm(sa2_first[..., element_idx] - goal[element_idx]) - np.linalg.norm(sa2_last[..., element_idx] - goal[element_idx])   

                        labels[i] =  (progress1 < progress2)
                    ###

                    # progress1 = 0. ; progress2 = 0.
                    # for task in task_list:
                    #     element_idx = OBS_ELEMENT_INDICES[task]
                    #     goal = np.zeros_like(sa1_last) 
                    #     goal[element_idx] = OBS_ELEMENT_GOALS[task]
                    #     progress1 += np.linalg.norm(sa1_first[..., element_idx] - goal[element_idx]) - np.linalg.norm(sa1_last[..., element_idx] - goal[element_idx])
                    #     progress2 += np.linalg.norm(sa2_first[..., element_idx] - goal[element_idx]) - np.linalg.norm(sa2_last[..., element_idx] - goal[element_idx])
                    # labels[i] = (progress1 < progress2)

                    ### uncomment this for only microwave
                    # sa1_last = sa_t_1[i, -1]; sa2_last = sa_t_2[i, -1] 
                    # sa1_first = sa_t_1[i, -1]; sa2_first = sa_t_2[i, -1] 
                    # labels[i] = (sa1_first[22] - sa1_last[22]) < (sa2_first[22] - sa2_last[22])
                    
                    # waypoint reward function
                    # sa1 = sa_t_1[i, -1]; sa2 = sa_t_2[i, -1] 
                    # t1, dist1 = get_next_target(sa1); t2, dist2 = get_next_target(sa2)
                    # if t1 != t2:
                    #     labels[i] = (t1 < t2)
                    # else:
                    #     labels[i] = (dist1 > dist2)

                    # only microwave
                    # sa1 = sa_t_1[i, -1]; sa2 = sa_t_2[i, -1] #  obs_dim+act_dim, get the final state
                    # if sa1[22] > THRESHOLD_MICROWAVE or sa2[22] > THRESHOLD_MICROWAVE: # microwave open a litte?
                    #     labels[i] = 1*(sa1[22] < sa2[22])
                    # else:
                    #     labels[i] = 1*(((sa1[:9] - BEGIN_MICROWAVE) ** 2).sum() < ((sa2[:9] - BEGIN_MICROWAVE) ** 2).sum())

        elif self.teacher_type == 1: # noisy teacher
            labels = 1*(sum_r_t_1 < sum_r_t_2)
            adv_labels = 1*(sum_r_t_1 > sum_r_t_2)
            
            # index of elements where diff or reward < margin
            margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_margin).reshape(-1)
            margin_index = np.array(list(range(labels.shape[0])))[margin_index]
            len_labels = margin_index.shape[0]
            random_index = np.random.permutation(len_labels)
            margin_index = margin_index[random_index]
            # among them, change the labels with prob \epsilon
            noise_frac = int(len_labels*self.teacher_noise)
            noise_index = margin_index[:noise_frac]
            labels[noise_index] = adv_labels[noise_index]
            num_noise = len(noise_index)
        elif self.teacher_type == 2: # margin teacher
            labels = 1*(sum_r_t_1 < sum_r_t_2)
            if first_flag == 0:
                margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
                labels = labels[margin_index]
                sa_t_1 = sa_t_1[margin_index]
                sa_t_2 = sa_t_2[margin_index]
                r_t_1 = r_t_1[margin_index]
                r_t_2 = r_t_2[margin_index]

        elif self.teacher_type == 3: # strict teacher
            labels = 1*(sum_r_t_1 < sum_r_t_2)
            if first_flag == 0:
                max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
                max_index = (max_r_t > self.teacher_thres).reshape(-1)
                labels = labels[max_index]
                sa_t_1 = sa_t_1[max_index]
                sa_t_2 = sa_t_2[max_index]
                r_t_1 = r_t_1[max_index]
                r_t_2 = r_t_2[max_index]

        elif self.teacher_type == 4: # noisy teacher ver 2: making mistake \propto 1/margin
            labels = 1*(sum_r_t_1 < sum_r_t_2)
            adv_labels = 1*(sum_r_t_1 > sum_r_t_2)
            # big noise --> less sensitive to margin diff
            margin = -self.teacher_noise * torch.Tensor(np.abs(sum_r_t_1 - sum_r_t_2))
            margin = 0.5 * torch.exp(margin)
            noise_index = torch.bernoulli(margin).numpy() > 0
            labels[noise_index] = adv_labels[noise_index]
            num_noise = np.sum(noise_index)
        elif self.teacher_type == 5: # noisy teacher ver 3: naive noisy teacher
            labels = 1*(sum_r_t_1 < sum_r_t_2)
            adv_labels = 1*(sum_r_t_1 > sum_r_t_2)
            
            len_labels = labels.shape[0]
            random_index = np.random.permutation(len_labels)
            noise_frac = int(len_labels*self.teacher_noise)
            noise_index = random_index[:noise_frac]
            labels[noise_index] = adv_labels[noise_index]
            num_noise = np.sum(noise_index)
        elif self.teacher_type == 6: # potential seeking
            labels = self.get_potential_label(r_t_1, r_t_2)
            # looks like not effective --> also can make some credit assignment issue...
        elif self.teacher_type == 7: # margin teacher
            labels = 1*(sum_r_t_1 < sum_r_t_2)
            margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
            labels = labels[margin_index]
            sa_t_1 = sa_t_1[margin_index]
            sa_t_2 = sa_t_2[margin_index]
            r_t_1 = r_t_1[margin_index]
            r_t_2 = r_t_2[margin_index]
            
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise
    
    def get_potential_label(self, r_t_1, r_t_2):
        margin = int(self.teacher_margin)
        total_len = int(self.size_segment + 1 - margin)
        for index in range(total_len):
            seg_reward_1 = np.sum(r_t_1[:, index:index+margin], axis=1)
            seg_reward_2 = np.sum(r_t_2[:, index:index+margin], axis=1)
            if index == 0:
                max_reward_1 = seg_reward_1
                max_reward_2 = seg_reward_2
            else:
                max_reward_1 = np.maximum(max_reward_1, seg_reward_1)
                max_reward_2 = np.maximum(max_reward_2, seg_reward_2)
        label = 1*(max_reward_1 < max_reward_2)
        return label
    
    def update_best_samples(self, sa_t_1, sa_t_2, r_t_1, r_t_2, labels):
        
        batch1_index = (labels == 0).reshape(-1)
        batch2_index = (labels > 0).reshape(-1)
        num_zeros = sum(batch1_index)
        
        # corner case 0: all labels = 1
        if num_zeros == 0:
            self.best_seg = sa_t_2
            self.best_label = r_t_2

        # corner case 1: all labels = 0
        elif num_zeros == self.mb_size:
            self.best_seg = sa_t_1
            self.best_label = r_t_1
        else:
            self.best_seg = np.concatenate([sa_t_1[batch1_index], sa_t_2[batch2_index]])
            self.best_label = np.concatenate([r_t_1[batch1_index], r_t_2[batch2_index]])
            
    def uniform_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=self.mb_size, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=self.mb_size, 
                img_flag=False, 
                recent_flag=recent_flag)
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def kcenter_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        
        num_init = self.mb_size*self.large_batch
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        
        # get final queries based on kmeans clustering
        total_sa = np.concatenate([sa_t_1.reshape(num_init, -1), 
                                   sa_t_2.reshape(num_init, -1)], axis=1)
        temp_len = total_sa.shape[0]
        feature_dim = total_sa.shape[1]
        X_inputs = total_sa.reshape(-1, feature_dim)
        
        center_index = KMeans(torch.Tensor(X_inputs).to(device), K=self.mb_size)
        center_index = center_index.data.cpu().numpy()
        r_t_1, sa_t_1 = r_t_1[center_index], sa_t_1[center_index]
        r_t_2, sa_t_2 = r_t_2[center_index], sa_t_2[center_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def kcenter_disagree_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        total_sa = np.concatenate([sa_t_1.reshape(num_init_half, -1), 
                                   sa_t_2.reshape(num_init_half, -1)], axis=1)
        temp_len = total_sa.shape[0]
        feature_dim = total_sa.shape[1]
        X_inputs = total_sa.reshape(-1, feature_dim)
        
        center_index = KMeans(torch.Tensor(X_inputs).to(device), K=self.mb_size)
        center_index = center_index.data.cpu().numpy()
        r_t_1, sa_t_1 = r_t_1[center_index], sa_t_1[center_index]
        r_t_2, sa_t_2 = r_t_2[center_index], sa_t_2[center_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def kcenter_entropy_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        
        num_init = self.mb_size*self.large_batch
        num_init_half = int(num_init*0.5)
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:num_init_half]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get final queries based on kmeans clustering
        total_sa = np.concatenate([sa_t_1.reshape(num_init_half, -1), 
                                   sa_t_2.reshape(num_init_half, -1)], axis=1)
        temp_len = total_sa.shape[0]
        feature_dim = total_sa.shape[1]
        X_inputs = total_sa.reshape(-1, feature_dim)
        
        center_index = KMeans(torch.Tensor(X_inputs).to(device), K=self.mb_size)
        center_index = center_index.data.cpu().numpy()
        r_t_1, sa_t_1 = r_t_1[center_index], sa_t_1[center_index]
        r_t_2, sa_t_2 = r_t_2[center_index], sa_t_2[center_index]

        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def kcenter_wo_act_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        
        num_init = self.mb_size*self.large_batch
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=num_init, 
                img_flag=False, 
                recent_flag=recent_flag)
        
        # get final queries based on kmeans clustering
        temp_sa_t_1 = sa_t_1[:,:,:self.ds]
        temp_sa_t_2 = sa_t_2[:,:,:self.ds]

        total_sa = np.concatenate([temp_sa_t_1.reshape(num_init, -1), 
                                   temp_sa_t_2.reshape(num_init, -1)], axis=1)
        temp_len = total_sa.shape[0]
        feature_dim = total_sa.shape[1]
        X_inputs = total_sa.reshape(-1, feature_dim)
        
        center_index = KMeans(torch.Tensor(X_inputs).to(device), K=self.mb_size)
        center_index = center_index.data.cpu().numpy()
        r_t_1, sa_t_1 = r_t_1[center_index], sa_t_1[center_index]
        r_t_2, sa_t_2 = r_t_2[center_index], sa_t_2[center_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def disagreement_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        disagree = disagree[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
        for _index in range(len(margin_index)):
            self.f_io.write("{}, {}, {}\n".format(
                self.round_counter, disagree[_index], margin_index[_index]))
        self.round_counter += 1
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def disagreement_exploit_sampling(self, recent_flag=False, first_flag=0):
        
        # get queries
        if len(self.best_label) == 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            total_num_samples = self.mb_size*self.large_batch
            num_new_samples = 2*total_num_samples - len(self.best_label)
            sa_t_1, _, r_t_1, _ =  self.get_queries(
                mb_size=num_new_samples, 
                img_flag=False, 
                recent_flag=recent_flag)
            
            total_sa_t = np.concatenate((sa_t_1, self.best_seg), axis=0)
            total_r_t = np.concatenate((r_t_1, self.best_label), axis=0)
            
            random_index = np.random.permutation(2*total_num_samples)
            
            sa_t_1 = total_sa_t[random_index[:total_num_samples]]
            r_t_1 = total_r_t[random_index[:total_num_samples]]
            sa_t_2 = total_sa_t[random_index[total_num_samples:]]
            r_t_2 = total_r_t[random_index[total_num_samples:]]
            
        # get final queries based on uncertainty
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        disagree = disagree[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
        for _index in range(len(margin_index)):
            self.f_io.write("{}, {}, {}\n".format(
                self.round_counter, disagree[_index], margin_index[_index]))
        self.round_counter += 1
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
                sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
            self.update_best_samples(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
                
        return len(labels), num_noise
    
    def entropy_exploit_sampling(self, recent_flag=False, first_flag=0):
        
        # get queries
        if len(self.best_label) == 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            total_num_samples = self.mb_size*self.large_batch
            num_new_samples = 2*total_num_samples - len(self.best_label)
            sa_t_1, _, r_t_1, _ =  self.get_queries(
                mb_size=num_new_samples, 
                img_flag=False, 
                recent_flag=recent_flag)
            
            total_sa_t = np.concatenate((sa_t_1, self.best_seg), axis=0)
            total_r_t = np.concatenate((r_t_1, self.best_label), axis=0)
            
            random_index = np.random.permutation(2*total_num_samples)
            
            sa_t_1 = total_sa_t[random_index[:total_num_samples]]
            r_t_1 = total_r_t[random_index[:total_num_samples]]
            sa_t_2 = total_sa_t[random_index[total_num_samples:]]
            r_t_2 = total_r_t[random_index[total_num_samples:]]
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        entropy = entropy[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
        for _index in range(len(margin_index)):
            self.f_io.write("{}, {}, {}\n".format(
                self.round_counter, entropy[_index], margin_index[_index]))
        self.round_counter += 1
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
            self.update_best_samples(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
            
        return len(labels), num_noise
    
    def oracle_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        
        # get final queries based on reward diff
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)

        max_r_t = np.abs(sum_r_t_1 - sum_r_t_2).reshape(-1)
        top_k_index = (max_r_t).argsort()[:self.mb_size]

        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label( 
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def entropy_sampling(self, recent_flag=False, first_flag=0, seg_margin=0):
        
        # get queries
        if seg_margin > 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries_one_seg(
                seg_margin=seg_margin,
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        else:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=self.mb_size*self.large_batch, 
                img_flag=False, 
                recent_flag=recent_flag)
        
        # get final queries based on uncertainty
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        entropy = entropy[top_k_index]
        
        # logging
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) > self.teacher_margin).reshape(-1)
        for _index in range(len(margin_index)):
            self.f_io.write("{}, {}, {}\n".format(
                self.round_counter, entropy[_index], margin_index[_index]))
        self.round_counter += 1
        
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(    
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def exploit_sampling(self, recent_flag=False, first_flag=0):
        
        # get queries
        if len(self.best_label) == 0:
            sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
                mb_size=self.mb_size, img_flag=False, recent_flag=recent_flag)
        else:
            num_new_samples = 2*self.mb_size - len(self.best_label)
            sa_t_1, _, r_t_1, _ =  self.get_queries(
                mb_size=num_new_samples, img_flag=False, recent_flag=recent_flag)

            total_sa_t = np.concatenate((sa_t_1, self.best_seg), axis=0)
            total_r_t = np.concatenate((r_t_1, self.best_label), axis=0)
            
            random_index = np.random.permutation(2*self.mb_size)
            
            sa_t_1 = total_sa_t[random_index[:self.mb_size]]
            r_t_1 = total_r_t[random_index[:self.mb_size]]
            sa_t_2 = total_sa_t[random_index[self.mb_size:]]
            r_t_2 = total_r_t[random_index[self.mb_size:]]
            
        # get labels
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(
            sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
            self.update_best_samples(sa_t_1, sa_t_2, r_t_1, r_t_2, labels)
        
        return len(labels), num_noise

    def knockout_sampling(self, recent_flag=False, first_flag=0):
        
        # get queries
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(
            mb_size=self.mb_size*self.large_batch, img_flag=False, recent_flag=recent_flag)
        
        # only use sa_t_1
        # let's pair them up first
        n = len(sa_t_1)
        sa_t_1_b, sa_t_2_b = np.repeat(sa_t_1[None, ...], n, axis=0), np.repeat(sa_t_1[:, None, ...], n, axis=1)
       
        # boolean matrix that indicates whether the first is clearly better than the second
        dominance_mat = []
        for _ in range(n):
            dominance_mat.append(self.get_dominance(sa_t_1_b[_], sa_t_2_b[_]))
        dominance_mat = np.stack(dominance_mat, axis=0)
        
        assert len(np.shape(dominance_mat)) == 2
        
        score = np.mean(dominance_mat, axis=0)    
        arg = np.argsort(score)
        
        sa_t_1_sort = sa_t_1[arg]
        r_t_1_sort = r_t_1[arg]
        
        remain = sa_t_1_sort[:self.mb_size * 2]
        remain_r = r_t_1_sort[:self.mb_size * 2]
        
        # randomly pair them up get labels
        sa_t_1, sa_t_2 = remain[:self.mb_size], remain[self.mb_size:]
        r_t_1, r_t_2 = remain_r[:self.mb_size], remain_r[self.mb_size:]

        sa_t_1, sa_t_2, r_t_1, r_t_2, labels, num_noise = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2, first_flag=first_flag)
        
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        
        return len(labels), num_noise
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])
        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0
        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0
            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len
                
            for member in range(self.de):
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                # get logits
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)

                # compute loss
                curr_loss = self.CEloss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.opt.step()
        
        ensemble_acc = ensemble_acc / total
        
        return ensemble_losses, list_debug_loss1, list_debug_loss2, ensemble_acc

    # added
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.de)]
        ensemble_acc = np.array([0 for _ in range(self.de)])        
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.de):
            total_batch_index.append(np.random.permutation(max_len))        
            num_epochs = int(np.ceil(max_len/self.train_batch_size))
        list_debug_loss1, list_debug_loss2 = [], []
        total = 0        
        for epoch in range(num_epochs):
            self.opt.zero_grad()
            loss = 0.0            
            last_index = (epoch+1)*self.train_batch_size
            if last_index > max_len:
                last_index = max_len            
            for member in range(self.de):                
                
                # get random batch
                idxs = total_batch_index[member][epoch*self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs]
                sa_t_2 = self.buffer_seg2[idxs]
                labels = self.buffer_label[idxs]
                labels = torch.from_numpy(labels.flatten()).float().to(device)  # change: .long() -> .float()

                if member == 0:
                    total += labels.size(0)                # get logits
                
                r_hat1 = self.r_hat_member(sa_t_1, member=member)
                r_hat2 = self.r_hat_member(sa_t_2, member=member)
                r_hat1 = r_hat1.sum(axis=1)
                r_hat2 = r_hat2.sum(axis=1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1)                
                
                # compute loss
                # uniform_index = labels == -1
                # labels[uniform_index] = 0
                # target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                # target_onehot += self.label_margin
                # if sum(uniform_index) > 0:
                #     target_onehot[uniform_index] = 0.5
                onehot_labels = labels.unsqueeze(1)
                onehot_labels = torch.cat([onehot_labels, 1 - onehot_labels], dim=-1)
                curr_loss = self.softXEnt_loss(r_hat, onehot_labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())                

                # cast labels to long
                labels = (labels < 0.5).long()
                
                # compute acc
                _, predicted = torch.max(r_hat.data, 1)
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct            
            loss.backward()
            self.opt.step()        
        ensemble_acc = ensemble_acc / total        
        return ensemble_losses, list_debug_loss1, list_debug_loss2, ensemble_acc

    def softXEnt_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax (input, dim = 1)
        return  -(target * logprobs).sum() / input.shape[0]