import numpy as np
from collections import defaultdict
import d4rl

from spirl.utils.general_utils import AttrDict
from spirl.utils.general_utils import ParamDict
from spirl.rl.components.environment import GymEnv

from reward_classifier import RNNRewardPredictor
import torch
from collections import deque

# class KitchenEnv(GymEnv):
#     """Tiny wrapper around GymEnv for Kitchen tasks."""
#     SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']
#     def init_rew_classifer(self):
#         self.reward_classifier = RNNRewardPredictor(60,9, 200, 1e-4, pos_weight=torch.ones((1,)).cuda()).cuda()
#         self.reward_classifier.load_state_dict(torch.load('rnn_reward_model/epoch1000'))
#         self.reward_classifier.eval()
#         self._k = 98
#         self._obses = deque([], maxlen=self._k)

#     def _default_hparams(self):
#         return super()._default_hparams().overwrite(ParamDict({
#             'name': "kitchen-mixed-v0",
#         }))

#     def step(self, *args, **kwargs):
#         obs, rew, done, info = super().step(*args, **kwargs)

#         task_onehot = torch.zeros((4,))
#         task_onehot.scatter_(0, torch.as_tensor(rew).long(), 1) 
#         _obs = torch.cat((torch.as_tensor(obs), torch.as_tensor(args[0]), task_onehot), dim=-1).cpu().numpy()

#         self._obses.append(_obs)
#         traj = np.concatenate(list(self._obses), axis=0).reshape(-1, self._obses[0].shape[0])
#         r_pred = float(self.reward_classifier.predict(traj)[-1] > 0.5)

#         return obs, np.float64(r_pred), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

#     def reset(self):
#         self.solved_subtasks = defaultdict(lambda: 0)
#         obs = super().reset()

#         _obs = np.concatenate((obs, np.asarray([0.,0.,0.,0.,0.,0.,0.,0.,0.,1., 0., 0., 0.])), axis=-1)
#         for _ in range(self._k):
#             self._obses.append(_obs)
#         return obs

#     def get_episode_info(self):
#         info = super().get_episode_info()
#         info.update(AttrDict(self.solved_subtasks))
#         return info

#     def _postprocess_info(self, info):
#         """Sorts solved subtasks into separately logged elements."""
#         completed_subtasks = info['completed_tasks'] #info.pop("completed_tasks") # change
#         for task in self.SUBTASKS:
#             self.solved_subtasks[task] = 1 if task in completed_subtasks or self.solved_subtasks[task] else 0
#         return info



class KitchenEnv(GymEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    SUBTASKS = ['microwave', 'kettle', 'slide cabinet', 'hinge cabinet', 'bottom burner', 'light switch', 'top burner']
    def _default_hparams(self):
        return super()._default_hparams().overwrite(ParamDict({
            'name': "kitchen-mixed-v0",
        }))

    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        return obs, np.float64(rew), done, self._postprocess_info(info)     # casting reward to float64 is important for getting shape later

    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        return super().reset()

    def get_episode_info(self):
        info = super().get_episode_info()
        info.update(AttrDict(self.solved_subtasks))
        return info

    def _postprocess_info(self, info):
        """Sorts solved subtasks into separately logged elements."""
        completed_subtasks = info["completed_tasks"]
        for task in self.SUBTASKS:
            self.solved_subtasks[task] = 1 if task in completed_subtasks or self.solved_subtasks[task] else 0
        return info

class NoGoalKitchenEnv(KitchenEnv):
    """Splits off goal from obs."""
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        obs = obs[:int(obs.shape[0]/2)]
        return obs, rew, done, info

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        return obs[:int(obs.shape[0]/2)]


class FinishMicrowaveKitchenEnv(KitchenEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    def reset(self):
        self._env.init_qpos[22] = -0.75
        obs = super().reset()
        self._env.tasks_to_complete.remove('microwave')
        return obs

### one task env ###
class MicrowaveKitchenEnv(KitchenEnv):
    """Tiny wrapper around GymEnv for Kitchen tasks."""
    def reset(self):
        self.solved_subtasks = defaultdict(lambda: 0)
        self.all_finish = False
        self.steps = 0
        self.max_step = 100
        return super().reset()
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        rew = ("microwave" in info["completed_tasks"]) and not self.all_finish
        rew = float(rew)
        if rew > 0:
            self.all_finish = True
        self.steps += 1
        done = done or (self.steps >= self.max_step)
        return obs, np.float64(rew), np.bool_(done), self._postprocess_info(info) 


class KettleKitchenEnv(KitchenEnv):
    def reset(self):
        self._env.TASK_ELEMENTS  = ['kettle', 'light switch']
        self.solved_subtasks = defaultdict(lambda: 0)
        self.steps = 0
        self.max_step = 100 * (len(self._env.TASK_ELEMENTS) - 1)
        return super().reset()
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        self.steps += 1
        done = done or (self.steps >= self.max_step)
        return obs, np.float64(rew), np.bool_(done), self._postprocess_info(info) 

### two task env ###
class MicrowaveKettleKitchenEnv(KitchenEnv):
    def reset(self):
        self._env.TASK_ELEMENTS  = ['microwave', 'kettle', 'light switch']
        self.solved_subtasks = defaultdict(lambda: 0)
        self.steps = 0
        self.max_step = 100 * (len(self._env.TASK_ELEMENTS) - 1)
        return super().reset()
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        self.steps += 1
        done = done or (self.steps >= self.max_step)
        return obs, np.float64(rew), np.bool_(done), self._postprocess_info(info) 


class KettleBurnerKitchenEnv(KitchenEnv):
    def reset(self):
        self._env.TASK_ELEMENTS  = ['kettle',  'bottom burner', 'light switch']
        self.solved_subtasks = defaultdict(lambda: 0)
        self.max_step = 100 * (len(self._env.TASK_ELEMENTS) - 1)
        self.steps = 0
        return super().reset()
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        self.steps += 1
        done = done or (self.steps >= self.max_step)
        return obs, np.float64(rew), np.bool_(done), self._postprocess_info(info) 

### three task env ###
class KettleBurnerCabinetKitchenEnv(KitchenEnv):
    def reset(self):
        self._env.TASK_ELEMENTS  = ['kettle',  'bottom burner', 'slide cabinet', 'light switch']
        self.solved_subtasks = defaultdict(lambda: 0)
        self.max_step = 280
        self.steps = 0
        return super().reset()
    
    def step(self, *args, **kwargs):
        obs, rew, done, info = super().step(*args, **kwargs)
        self.steps += 1
        done = done or (self.steps >= self.max_step)
        return obs, np.float64(rew), np.bool_(done), self._postprocess_info(info) 



