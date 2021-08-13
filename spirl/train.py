import matplotlib
from torch.utils.data import dataloader; matplotlib.use('Agg')
import torch
import torch.nn as nn
import os
import time
from shutil import copy
import datetime
import imp
from tensorboardX import SummaryWriter
import numpy as np
import random
from torch import autograd
from torch.optim import Adam, RMSprop, SGD
from functools import partial

from spirl.components.data_loader import RandomVideoDataset
from spirl.utils.general_utils import RecursiveAverageMeter, map_dict
from spirl.components.checkpointer import CheckpointHandler, save_cmd, save_git, get_config_path
from spirl.utils.general_utils import dummy_context, AttrDict, get_clipped_optimizer, \
                                                        AverageMeter, ParamDict
from spirl.utils.pytorch_utils import LossSpikeHook, NanGradHook, NoneGradHook, \
                                                        DataParallelWrapper, RAdam
from spirl.components.trainer_base import BaseTrainer
from spirl.components.params import get_args

# change
from functools import reduce
import wandb 
import spirl

class ModelTrainer(BaseTrainer):
    def __init__(self, args):
        
        wandb.init(project="skill_learning")# change

        self.args = args
        self.setup_device()

        # set up params
        self.conf = conf = self.get_config()

        self._hp = self._default_hparams()
        self._hp.overwrite(conf.general)  # override defaults with config file
        self._hp.exp_path = make_path(conf.exp_dir, args.path, args.prefix, args.new_dir)
        self.log_dir = log_dir = os.path.join(self._hp.exp_path, 'events')
        print('using log dir: ', log_dir)
        self.conf = self.postprocess_conf(conf)
        if args.deterministic: set_seeds()

        # set up logging + training monitoring
        self.writer = self.setup_logging(conf, self.log_dir)
        self.setup_training_monitors()
        
        # buld dataset, model. logger, etc.
        train_params = AttrDict(logger_class=self._hp.logger,
                                model_class=self._hp.model,
                                n_repeat=self._hp.epoch_cycles_train,
                                dataset_size=-1)
        self.logger, self.model, self.train_loader = self.build_phase(train_params, 'train')

        test_params = AttrDict(logger_class=self._hp.logger if self._hp.logger_test is None else self._hp.logger_test,
                               model_class=self._hp.model if self._hp.model_test is None else self._hp.model_test,
                               n_repeat=1,
                               dataset_size=args.val_data_size)
        self.logger_test, self.model_test, self.val_loader = self.build_phase(test_params, phase='val')

        # set up optimizer + evaluator
        self.optimizer = self.get_optimizer_class()(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self._hp.lr)
        self.evaluator = self._hp.evaluator(self._hp, self.log_dir, self._hp.top_of_n_eval,
                                            self._hp.top_comp_metric, tb_logger=self.logger_test)

        # setup offline dataset: optimal vs noisy
        if not self.args.mixed_data:
            self.conf.data.dataset_spec.dataset_class = spirl.data.kitchen.src.kitchen_data_loader.D4RLSequenceSplitDataset

        # set up classifier for random vs expert trajectory, change
        self.ensemble = True
        lr =  1e-3
        weight_decay = 1e-4
        if self.ensemble:
            self.ensemble = []
            params = []
            for _ in range(3):
                model = torch.nn.Sequential(nn.Linear(self.conf.model.n_rollout_steps * (self.conf.model.state_dim + self.conf.model.action_dim)+self.conf.model.state_dim, 200),
                                                nn.ReLU(),
                                                nn.Linear(200,200), 
                                                nn.ReLU(),
                                                nn.Linear(200, 1)).cuda()
                self.ensemble.append(model)
                params.extend(model.parameters())
            self.gen_human_opt = Adam(params, lr=lr, weight_decay=weight_decay)
            
                
        else:
            self.generalized_human = torch.nn.Sequential(nn.Linear(self.conf.model.n_rollout_steps * (self.conf.model.state_dim + self.conf.model.action_dim)+self.conf.model.state_dim, 200),
                                                nn.ReLU(),
                                                nn.Linear(200,200), 
                                                nn.ReLU(),
                                                nn.Linear(200, 1)).cuda()
            self.gen_human_opt = Adam(self.generalized_human.parameters(), lr=lr, weight_decay=weight_decay)
        self.gen_human_loss = torch.nn.BCEWithLogitsLoss()
        self.human_loader = self.get_dataset(self.args, AttrDict(resolution=None), self.conf.data, 'human', train_params.n_repeat, -1)

        
        

        
        # load model params from checkpoint
        self.global_step, start_epoch = 0, 0
        if args.resume or conf.ckpt_path is not None:
            start_epoch = self.resume(args.resume, conf.ckpt_path)

        if args.val_sweep:
            self.run_val_sweep()
        elif args.train:
            self.train(start_epoch)
        else:
            self.val()


    ### inside of the generalized human model, change
    def forward_gen_human(self, states, actions):
        batch = states.shape[0]
        states = states.reshape(batch, -1)
        actions = actions.reshape(batch, -1)
        inputs = torch.cat((states, actions), -1)
        return self.generalized_human(inputs)

    def train_gen_human(self, states, actions, labels):
        self.generalized_human.train()
        labels = labels.unsqueeze(-1)
        pred = self.forward_gen_human(states, actions)
        loss = self.gen_human_loss(pred, labels)
        self.gen_human_opt.zero_grad()
        loss.backward()
        self.gen_human_opt.step()
        # log train loss here
        wandb.log({'train loss': loss.item()})

        pred = (torch.sigmoid(pred) > 0.5).float()
        train_acc = 1 - torch.sum(torch.abs(pred.squeeze() - labels.squeeze())) / pred.shape[0]
        wandb.log({'train accuracy': train_acc.item()})
        return train_acc
    
    def predict_gen_human(self, states, actions, binary=True):
        self.generalized_human.eval()
        with torch.no_grad():
            pred = self.forward_gen_human(states, actions)
        if binary:
            return (torch.sigmoid(pred) > 0.5).float()
        else:
            return torch.sigmoid(pred).detach()

    ## an ensemble
    def forward_gen_human_ensemble(self, states, actions, index=0):
        batch = states.shape[0]
        states = states.reshape(batch, -1)
        actions = actions.reshape(batch, -1)
        inputs = torch.cat((states, actions), -1)
        return self.ensemble[index](inputs)

    def train_gen_human_ensemble(self, states, actions, labels):
        loss = 0
        for i in range(3):
            model = self.ensemble[i]
            model.train()
            pred = self.forward_gen_human_ensemble(states, actions, index=i)
            loss = loss + self.gen_human_loss(pred, labels.unsqueeze(-1))
        self.gen_human_opt.zero_grad()
        loss.backward()
        self.gen_human_opt.step()

        wandb.log({'train loss': loss.item()/3})
    
    def predict_gen_human_ensemble(self, states, actions, binary=True):
        total = None
        for i in range(3):
            self.ensemble[i].eval()
            with torch.no_grad():
                pred = self.forward_gen_human_ensemble(states, actions, i)
                if total is None:
                    total = pred
                else:
                    total += pred 

        pred = total / 3.
        if binary:
            return (torch.sigmoid(pred) > 0.5).float()
        else:
            return torch.sigmoid(pred).detach()
    ###
    
    def _default_hparams(self):
        default_dict = ParamDict({
            'model': None,
            'model_test': None,
            'logger': None,
            'logger_test': None,
            'evaluator': None,
            'data_dir': None,  # directory where dataset is in
            'batch_size': 16,
            'exp_path': None,  # Path to the folder with experiments
            'num_epochs': 200,
            'epoch_cycles_train': 1,
            'optimizer': 'radam',    # supported: 'adam', 'radam', 'rmsprop', 'sgd'
            'lr': 1e-3,
            'gradient_clip': None,
            'momentum': 0,      # momentum in RMSProp / SGD optimizer
            'adam_beta': 0.9,       # beta1 param in Adam
            'top_of_n_eval': 1,     # number of samples used at eval time
            'top_comp_metric': None,    # metric that is used for comparison at eval time (e.g. 'mse')
        })
        return default_dict
    
    def train(self, start_epoch):
        # if not self.args.skip_first_val: #change
        #     self.val()
        self.train_soft_weight()
        for epoch in range(start_epoch, self._hp.num_epochs):     
            self.train_epoch(epoch)
        
            if not self.args.dont_save:
                save_checkpoint({
                    'epoch': epoch,
                    'global_step': self.global_step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                },  os.path.join(self._hp.exp_path, 'weights'), CheckpointHandler.get_ckpt_name(epoch))

            if epoch % self.args.val_interval == 0:
                self.val()

    def train_soft_weight(self):
         ## separate the train of learned generatlized human here
        if self.ensemble:
            for epoch in range(self.args.train_epoch_from_human):
                for batch_idx, sample_batched in enumerate(self.human_loader):
                    inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                    self.train_gen_human_ensemble(inputs['states'], inputs['actions'], inputs['labels'])
                    

                sample_batched = next(iter(self.train_loader))
                inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                pred_prob = self.predict_gen_human_ensemble(inputs['states'], inputs['actions'], False).squeeze()
                val_acc =  1 - torch.sum(torch.abs((pred_prob>0.5).float()-inputs['labels'])) / pred_prob.shape[0]
                wandb.log({'validation accuracy': val_acc.item()})
                val_dif = torch.sum(torch.abs(pred_prob-inputs['labels']))
                wandb.log({'validation difference': val_dif.item()})

        else:
            for epoch in range(self.args.train_epoch_from_human):
                for batch_idx, sample_batched in enumerate(self.human_loader):
                    inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                    train_acc = self.train_gen_human(inputs['states'], inputs['actions'], inputs['labels'])
                    

                sample_batched = next(iter(self.train_loader))
                inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
                pred_prob = self.predict_gen_human(inputs['states'], inputs['actions'], False).squeeze()
                val_acc =  1 - torch.sum(torch.abs((pred_prob>0.5).float()-inputs['labels'])) / pred_prob.shape[0]
                wandb.log({'validation accuracy': val_acc.item()})
                val_dif = torch.sum(torch.abs(pred_prob-inputs['labels']))
                wandb.log({'validation difference': val_dif.item()})
        return # uncomment this line to only train the soft human generalizer 

    def train_epoch(self, epoch):
        self.model.train()
        epoch_len = len(self.train_loader)
        end = time.time()
        batch_time = AverageMeter()
        upto_log_time = AverageMeter()
        data_load_time = AverageMeter()
        self.log_outputs_interval = self.args.log_interval
        self.log_images_interval = int(epoch_len / self.args.per_epoch_img_logs)
        
        print('starting epoch ', epoch)

        for self.batch_idx, sample_batched in enumerate(self.train_loader):
            data_load_time.update(time.time() - end)
            inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))
            ### change
            soft_train = self.args.soft_train
            batch_size = inputs['states'].shape[0]
            if not soft_train:
                pred_mask = self.predict_gen_human(inputs['states'], inputs['actions'])
                wandb.log({'train overfit loss': torch.sum((inputs['labels']-pred_mask.squeeze())**2)})

                pred_mask = pred_mask.bool() #inputs['labels'].unsqueeze(1).bool() #
                if torch.sum(pred_mask) < 3:
                    continue
                
                for _ in range(2):
                    if torch.sum(pred_mask) >= batch_size:
                        break
                    another_batch = next(iter(self.train_loader))
                    another_inputs = AttrDict(map_dict(lambda x: x.to(self.device), another_batch))
                    for key in inputs.keys():
                        inputs[key] = torch.cat((inputs[key], another_inputs[key]), dim=0)
                    # pred_mask = inputs['labels'].unsqueeze(1).bool()
                    another_pred_mask = self.predict_gen_human(another_inputs['states'], another_inputs['actions']).bool()
                    pred_mask = torch.cat((pred_mask, another_pred_mask), dim=0)

                if torch.sum(pred_mask) < 16:
                    continue 
                
                for key in inputs.keys():
                    if key == 'labels':
                        continue
                    pred_mask_ = pred_mask
                    for _ in range(len(inputs[key].shape) - 2):
                        pred_mask_ = pred_mask_.unsqueeze(-1)
                    inputs[key] = torch.masked_select(inputs[key], pred_mask_.expand(*inputs[key].shape))[:16]
            else:
                if self.ensemble:
                    pred_weight = self.predict_gen_human_ensemble(inputs['states'], inputs['actions'], binary=False)
                else:
                    pred_weight = self.predict_gen_human(inputs['states'], inputs['actions'], binary=False)
            ###

            with self.training_context():
                self.optimizer.zero_grad()
                output = self.model(inputs)
                if soft_train:
                    losses = self.model.loss_soft(output, inputs, pred_weight)
                else:
                    losses = self.model.loss(output, inputs)
                losses.total.value.backward()
                self.call_hooks(inputs, output, losses, epoch)

                self.optimizer.step()
                self.model.step()

            if self.args.train_loop_pdb:
                import pdb; pdb.set_trace()
            
            upto_log_time.update(time.time() - end)
            if self.log_outputs_now and not self.args.dont_save:
                self.model.log_outputs(output, inputs, losses, self.global_step,
                                       log_images=self.log_images_now, phase='train', **self._logging_kwargs)
            batch_time.update(time.time() - end)
            end = time.time()
            
            if self.log_outputs_now:
                print('GPU {}: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"] if self.use_cuda else 'none',
                                          self._hp.exp_path))
                print(('itr: {} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.global_step, epoch, self.batch_idx, len(self.train_loader),
                        100. * self.batch_idx / len(self.train_loader), losses.total.value.item())))

                print('avg time for loading: {:.2f}s, logs: {:.2f}s, compute: {:.2f}s, total: {:.2f}s'
                      .format(data_load_time.avg,
                              batch_time.avg - upto_log_time.avg,
                              upto_log_time.avg - data_load_time.avg,
                              batch_time.avg))
                togo_train_time = batch_time.avg * (self._hp.num_epochs - epoch) * epoch_len / 3600.
                print('ETA: {:.2f}h'.format(togo_train_time))

            del output, losses
            self.global_step = self.global_step + 1

    def val(self):
        print('Running Testing')
        if self.args.test_prediction:
            start = time.time()
            self.model_test.load_state_dict(self.model.state_dict())
            losses_meter = RecursiveAverageMeter()
            self.model_test.eval()
            self.evaluator.reset()
            with autograd.no_grad():
                for batch_idx, sample_batched in enumerate(self.val_loader):
                    inputs = AttrDict(map_dict(lambda x: x.to(self.device), sample_batched))

                    # run evaluator with val-mode model
                    with self.model_test.val_mode():
                        self.evaluator.eval(inputs, self.model_test)

                    # run non-val-mode model (inference) to check overfitting
                    output = self.model_test(inputs)
                    losses = self.model_test.loss(output, inputs)

                    losses_meter.update(losses)
                    del losses
                
                if not self.args.dont_save:
                    if self.evaluator is not None:
                        self.evaluator.dump_results(self.global_step)

                    self.model_test.log_outputs(output, inputs, losses_meter.avg, self.global_step,
                                                log_images=True, phase='val', **self._logging_kwargs)
                    print(('\nTest set: Average loss: {:.4f} in {:.2f}s\n'
                           .format(losses_meter.avg.total.value.item(), time.time() - start)))
            del output

    def setup_device(self):
        self.use_cuda = torch.cuda.is_available() and not self.args.debug
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')
        if self.args.gpu != -1:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)

    def get_config(self):
        conf = AttrDict()

        # paths
        conf.exp_dir = self.get_exp_dir()
        conf.conf_path = get_config_path(self.args.path)

        # general and model configs
        print('loading from the config file {}'.format(conf.conf_path))
        conf_module = imp.load_source('conf', conf.conf_path)
        conf.general = conf_module.configuration
        conf.model = conf_module.model_config

        # data config
        try:
            data_conf = conf_module.data_config
        except AttributeError:
            data_conf_file = imp.load_source('dataset_spec', os.path.join(AttrDict(conf).data_dir, 'dataset_spec.py'))
            data_conf = AttrDict()
            data_conf.dataset_spec = AttrDict(data_conf_file.dataset_spec)
            data_conf.dataset_spec.split = AttrDict(data_conf.dataset_spec.split)
        conf.data = data_conf

        # model loading config
        conf.ckpt_path = conf.model.checkpt_path if 'checkpt_path' in conf.model else None

        return conf

    def postprocess_conf(self, conf):
        conf.model['batch_size'] = self._hp.batch_size if not torch.cuda.is_available() \
            else int(self._hp.batch_size / torch.cuda.device_count())
        conf.model.update(conf.data.dataset_spec)
        conf.model['device'] = conf.data['device'] = self.device.type
        return conf

    def setup_logging(self, conf, log_dir):
        if not self.args.dont_save:
            print('Writing to the experiment directory: {}'.format(self._hp.exp_path))
            if not os.path.exists(self._hp.exp_path):
                os.makedirs(self._hp.exp_path)
            save_cmd(self._hp.exp_path)
            save_git(self._hp.exp_path)
            save_config(conf.conf_path, os.path.join(self._hp.exp_path, "conf_" + datetime_str() + ".py"))
            writer = SummaryWriter(log_dir)
        else:
            writer = None

        # set up additional logging args
        self._logging_kwargs = AttrDict(
        )
        return writer

    def setup_training_monitors(self):
        self.training_context = autograd.detect_anomaly if self.args.detect_anomaly else dummy_context
        self.hooks = []
        self.hooks.append(LossSpikeHook('sg_img_mse_train'))
        self.hooks.append(NanGradHook(self))
        self.hooks.append(NoneGradHook(self))

    def build_phase(self, params, phase):
        if not self.args.dont_save:
            logger = params.logger_class(self.log_dir, summary_writer=self.writer)
        else:
            logger = None
        model = params.model_class(self.conf.model, logger)
        if torch.cuda.device_count() > 1:
            print("\nUsing {} GPUs!\n".format(torch.cuda.device_count()))
            model = DataParallelWrapper(model)
        model = model.to(self.device)
        model.device = self.device
        loader = self.get_dataset(self.args, model, self.conf.data, phase, params.n_repeat, params.dataset_size)
        return logger, model, loader

    def get_dataset(self, args, model, data_conf, phase, n_repeat, dataset_size=-1):
        if args.feed_random_data:
            dataset_class = RandomVideoDataset
        else:
            dataset_class = data_conf.dataset_spec.dataset_class 

        # change
        loader = dataset_class(self._hp.data_dir, data_conf, resolution=model.resolution,
                               phase=phase, shuffle=(phase == "train" or phase == "human"), dataset_size=dataset_size, learn_human_ratio=args.learn_human_ratio). \
            get_data_loader(self._hp.batch_size, n_repeat)

        return loader

    def resume(self, ckpt, path=None):
        path = os.path.join(self._hp.exp_path, 'weights') if path is None else os.path.join(path, 'weights')
        assert ckpt is not None  # need to specify resume epoch for loading checkpoint
        weights_file = CheckpointHandler.get_resume_ckpt_file(ckpt, path)
        self.global_step, start_epoch, _ = \
            CheckpointHandler.load_weights(weights_file, self.model,
                                           load_step=True, load_opt=True, optimizer=self.optimizer,
                                           strict=self.args.strict_weight_loading)
        self.model.to(self.model.device)
        return start_epoch

    def get_optimizer_class(self):
        optim = self._hp.optimizer
        if optim == 'adam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=Adam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'radam':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RAdam, betas=(self._hp.adam_beta, 0.999))
        elif optim == 'rmsprop':
            get_optim = partial(get_clipped_optimizer, optimizer_type=RMSprop, momentum=self._hp.momentum)
        elif optim == 'sgd':
            get_optim = partial(get_clipped_optimizer, optimizer_type=SGD, momentum=self._hp.momentum)
        else:
            raise ValueError("Optimizer '{}' not supported!".format(optim))
        return partial(get_optim, gradient_clip=self._hp.gradient_clip)

    def run_val_sweep(self):
        epochs = CheckpointHandler.get_epochs(os.path.join(self._hp.exp_path, 'weights'))
        for epoch in list(sorted(epochs))[::2]:
            self.resume(epoch)
            self.val()
        return

    def get_exp_dir(self):
        return os.environ['EXP_DIR']

    @property
    def log_images_now(self):
        return self.global_step % self.log_images_interval == 0

    @property
    def log_outputs_now(self):
        return self.global_step % self.log_outputs_interval == 0 or self.global_step % self.log_images_interval == 0


def save_checkpoint(state, folder, filename='checkpoint.pth'):
    os.makedirs(folder, exist_ok=True)
    torch.save(state, os.path.join(folder, filename))
    print(f"Saved checkpoint to {os.path.join(folder, filename)}!")


def get_exp_dir():
    return os.environ['EXP_DIR']


def datetime_str():
    return datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")


def make_path(exp_dir, conf_path, prefix, make_new_dir):
    # extract the subfolder structure from config path
    path = conf_path.split('configs/', 1)[1]
    if make_new_dir:
        prefix += datetime_str()
    base_path = os.path.join(exp_dir, path)
    return os.path.join(base_path, prefix) if prefix else base_path


def set_seeds(seed=0, cuda_deterministic=True):
    """Sets all seeds and disables non-determinism in cuDNN backend."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def save_config(conf_path, exp_conf_path):
    copy(conf_path, exp_conf_path)

        
if __name__ == '__main__':
    ModelTrainer(args=get_args())
