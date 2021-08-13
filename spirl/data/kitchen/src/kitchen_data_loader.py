import d4rl
import gym
import numpy as np
import itertools

from spirl.components.data_loader import Dataset
from spirl.utils.general_utils import AttrDict

import pickle

class D4RLSequenceSplitDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        env = gym.make(self.spec.env_name)
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            self.seqs = list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[self.spec.filter_indices[0] : self.spec.filter_indices[1]+1]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)


class D4RLAtomicDataset(D4RLSequenceSplitDataset):
    def __getitem__(self, index):
        # sample start index in data range
        seq = self._sample_seq()
        idx = np.random.randint(0, seq.states.shape[0])
        output = AttrDict(
            states=seq.states[idx],
            actions=seq.actions[idx],
            pad_mask=np.ones((self.subseq_len,)),
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output


class D4RLMixedDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, data_dir, data_conf, phase, resolution=None, shuffle=True, dataset_size=-1, learn_human_ratio=0.1):
        self.phase = phase
        self.data_dir = data_dir
        self.spec = data_conf.dataset_spec
        self.subseq_len = self.spec.subseq_len
        self.remove_goal = self.spec.remove_goal if 'remove_goal' in self.spec else False
        self.dataset_size = dataset_size
        self.device = data_conf.device
        self.n_worker = 4
        self.shuffle = shuffle

        env = gym.make(self.spec.env_name)
        self.dataset = env.get_dataset()

        # split dataset into sequences
        seq_end_idxs = np.where(self.dataset['terminals'])[0]
        start = 0
        self.seqs = []
        for end_idx in seq_end_idxs:
            if end_idx+1 - start < self.subseq_len: continue    # skip too short demos
            self.seqs.append(AttrDict(
                states=self.dataset['observations'][start:end_idx+1],
                actions=self.dataset['actions'][start:end_idx+1],
            ))
            start = end_idx+1

        self.last_expert_idx = len(self.seqs) # index of last expert sequence in self.seqs


        # add random actions into the dataset
        # hardcode the sequence length to be 200
        if phase == 'train' or phase == 'human':
            # seq_len = 200
            # add_size = len(self.seqs) 
            # for i in range(add_size):
            #     rand_obs = self.dataset['observations'][np.random.randint(0, len(self.dataset['observations']))]
            #     env.init_qpos[:30] = rand_obs[:30]
            #     obs = env.reset()
            #     obses = np.ones((seq_len, self.dataset['observations'][0].shape[0]), dtype=np.float32)
            #     acts = np.ones((seq_len, self.dataset['actions'][0].shape[0]), dtype=np.float32)
            #     for t in range(seq_len):
            #         act = env.action_space.sample()
            #         next_obs = env.step(act)[0]
            #         obses[t] = obs
            #         acts[t] = act
            #         obs = next_obs
            #     self.seqs.append(AttrDict(
            #         states=obses,
            #         actions=acts,
            #     ))
            #     if i % 10 == 0:
            #         print(i)
            # import pickle
            # l = self.seqs[-add_size:]
            # with open("random_action_dataset.txt", "wb") as fp:   #Pickling
            #     pickle.dump(l, fp)

            with open("random_action_dataset.txt", "rb") as fp:   # Unpickling
               b = pickle.load(fp)
            self.seqs.extend(b)

        # 0-pad sequences for skill-conditioned training
        if 'pad_n_steps' in self.spec and self.spec.pad_n_steps > 0:
            for seq in self.seqs:
                seq.states = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.states.shape[1]), dtype=seq.states.dtype), seq.states))
                seq.actions = np.concatenate((np.zeros((self.spec.pad_n_steps, seq.actions.shape[1]), dtype=seq.actions.dtype), seq.actions))

        # filter demonstration sequences
        if 'filter_indices' in self.spec:
            print("!!! Filtering kitchen demos in range {} !!!".format(self.spec.filter_indices))
            self.seqs = list(itertools.chain.from_iterable(itertools.repeat(x, self.spec.demo_repeats)
                               for x in self.seqs[self.spec.filter_indices[0] : self.spec.filter_indices[1]+1]))
            import random
            random.shuffle(self.seqs)

        self.n_seqs = len(self.seqs)

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)  
        elif self.phase == "human":
            mid_point = self.n_seqs//2
            margin = int(mid_point * learn_human_ratio)
            self.start = mid_point - margin
            self.end = mid_point + margin
            self.SPLIT['human'] = margin * 2 / self.n_seqs
            print("initializing human label dataset with ratio of", self.SPLIT['human'])
        else:
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs

    def __getitem__(self, index):
        # sample start index in data range
        # print(index)
        seq, seq_idx = self._sample_seq()
        start_idx = np.random.randint(0, seq.states.shape[0] - self.subseq_len - 1)
        output = AttrDict(
            states=seq.states[start_idx:start_idx+self.subseq_len],
            actions=seq.actions[start_idx:start_idx+self.subseq_len-1],
            pad_mask=np.ones((self.subseq_len,)),
            labels=float(seq_idx<self.last_expert_idx)
        )
        if self.remove_goal:
            output.states = output.states[..., :int(output.states.shape[-1]/2)]
        return output

    def _sample_seq(self):
        idx = np.random.randint(self.start, self.end)
        return self.seqs[idx], idx
        # return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        if self.dataset_size != -1:
            return self.dataset_size
        return int(self.SPLIT[self.phase] * self.dataset['observations'].shape[0] / self.subseq_len)
