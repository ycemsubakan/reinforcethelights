import argparse
import time
import datetime
import torch
import torch_ac
import sys
from gym import wrappers, logger

import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from abc import ABC, abstractmethod

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList, ParallelEnv
import numpy as np

sys.path.insert(0, "/home/user1/rl-starter-files")
import utils
import pdb
import matplotlib.pyplot as plt


def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, args, preprocess_obss, mode='train'):

        # Store parameters
        self.mode = mode
        if mode == 'train':
            self.env = (envs[0])  #ParallelEnv(envs) 
        else:
            self.env = (envs[0])
        self.envs = envs

        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = args.frames_per_proc
        self.discount = args.discount
        self.lr = args.lr
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = None
        self.render = args.render

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        
    def collect_experiences_basic(self):
        """Collects rollouts and computes advantages.
        Returns
        -------
        """

        all_obs = []
        all_actions = []
        all_rewards = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)
        log_probs = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)
        all_values = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)
        all_entropy = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)

        obs = self.env.reset()
        all_frames = []

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction
            
            if self.mode == 'train':  
                preprocessed_obs = self.preprocess_obss([obs], device=self.device)
            else:
                preprocessed_obs = self.preprocess_obss([obs], device=self.device)

            #with torch.no_grad():
            dist, value = self.acmodel(preprocessed_obs)
            action = dist.sample()
            #prbs = dist.probs
            #action = torch.argmax(prbs)

            #print(prbs, action)
            if self.render:
                self.env.render()
                time.sleep(0.1)
                all_frames.append(np.moveaxis(self.env.render("rgb_array"), 2, 0))

            obs, reward, done, _ = self.env.step(action.cpu().numpy())

            # Update experiences values
            all_obs.append(obs)
            all_actions.append(action)
            
            all_rewards[i] = torch.tensor(reward, device=self.device)
            log_probs[i] = dist.log_prob(action)
            all_entropy[i] = dist.entropy()
            all_values[i] = value

            
        # Preprocess experiences

        return {'obs': all_obs, 
                'actions': all_actions,
                'rewards': all_rewards, 
                'log_probs': log_probs, 
                'all_values': all_values,
                'all_frames': all_frames,
                'all_entropy': all_entropy}


    def collect_experiences_parallelfor(self):
        """Collects rollouts and computes advantages.
        Returns
        -------
        """

        all_obs = []
        all_actions = []
        all_rewards = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)
        log_probs = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)
        all_values = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)
        all_entropy = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)

        obs_par = [env.reset() for env in self.envs]
        all_frames = []

        for i in range(self.num_frames_per_proc):
            actions_par = []
            for j in range(self.num_procs):
                # Inner loop is for the episodes
            
                preprocessed_obs = self.preprocess_obss([obs_par[j]], device=self.device)

                #with torch.no_grad():
                dist, value = self.acmodel(preprocessed_obs)
                action = dist.sample()
                #prbs = dist.probs
                #action = torch.argmax(prbs)

                #print(prbs, action)
                if self.render:
                    self.env.render()
                    time.sleep(0.1)
                    all_frames.append(np.moveaxis(self.env.render("rgb_array"), 2, 0))

                obs, reward, done, _ = self.envs[j].step(action.cpu().numpy())

                # record the j'th episode
                obs_par[j] = obs
                actions_par.append(action)
                
                all_rewards[i, j] = reward
                log_probs[i, j] = dist.log_prob(action)
                all_entropy[i, j] = dist.entropy()
                all_values[i, j] = value

            all_actions.append(actions_par)
            all_obs.append(obs_par)

                
            # Preprocess experiences

        return {'obs': all_obs, 
                'actions': all_actions,
                'rewards': all_rewards, 
                'log_probs': log_probs, 
                'all_values': all_values,
                'all_frames': all_frames,
                'all_entropy': all_entropy}




    @abstractmethod
    def update_parameters(self):
        pass

class reinforce(BaseAlgo):
    """REINFORCE algorithm."""

    def __init__(self, envs, acmodel, device=None, args=None, preprocess_obss=None):

        super().__init__(envs, acmodel, device, args=args, preprocess_obss=preprocess_obss)

        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.acmodel.parameters(), args.lr)
        elif args.optimizer == 'Adam': 
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), args.lr)
        else:
            raise ValueError('Wrong optimizer name')

    def update_parameters(self, exps):
        
        gam = self.discount
        
        # epoints = (exps.mask.squeeze() == 0).nonzero()
        # T = epoints[0].item()

        T = exps['rewards'].shape[0]
        all_Gs = []
        for t in range(T):
            gams = ((torch.ones(T - t)*gam) ** torch.arange(T-t).float()).to(self.device)
            all_Gs.append((exps['rewards'][t:T] * gams).sum().to(self.device))

        all_Gs = torch.tensor(all_Gs).to(self.device) 
        
        self.optimizer.zero_grad()
        log_p = (- (all_Gs*exps['log_probs'][:T]).sum())
        print(log_p.item())
        log_p.backward(retain_graph=True)

        self.optimizer.step()

class reinforce_wbaseline(BaseAlgo):
    """reinforce algorithm with baseline"""

    def __init__(self, envs, acmodel, device=None, args=None, preprocess_obss=None):

        super().__init__(envs, acmodel, device, args=args, preprocess_obss=preprocess_obss)

        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.acmodel.parameters(), args.lr)
            self.valoptimizer = torch.optim.SGD(self.acmodel.critic.parameters(), args.lr)
        elif args.optimizer == 'Adam': 
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), args.lr)
            self.valoptimizer = torch.optim.Adam(self.acmodel.critic.parameters(), args.lr)
        else:
            raise ValueError('Wrong optimizer name')

    def update_parameters(self, exps, preprocess_obss=None):
        
        gam = self.discount
        
        # epoints = (exps.mask.squeeze() == 0).nonzero()
        # T = epoints[0].item()

        T = exps['rewards'].shape[0]
        all_Gs = []
        for t in range(T):
            gams = ((torch.ones(T - t)*gam) ** torch.arange(T-t).float()).to(self.device)
            all_Gs.append((exps['rewards'][t:T] * gams).sum().to(self.device))

        all_Gs = torch.tensor(all_Gs).to(self.device) 
        with torch.no_grad():
            all_deltas = (all_Gs- exps['all_values'])

        self.optimizer.zero_grad()

        log_p = (- (all_deltas*exps['log_probs'][:T]).mean())  
        print(log_p.item())
        log_p.backward(retain_graph=True)

        # set the critic grads to zero
        #self.acmodel.critic[0].weight.grad.data.fill_(0)
        #self.acmodel.critic[0].bias.grad.data.fill_(0)

        #self.acmodel.critic[2].weight.grad.data.fill_(0)
        #self.acmodel.critic[2].bias.grad.data.fill_(0)

        self.optimizer.step()

        # update the value parameters
        for nein in range(5):
            self.valoptimizer.zero_grad()
            valuepart = ((all_Gs - exps['all_values'])**2).mean()
            valuepart.backward(retain_graph=True)
            self.valoptimizer.step()

        #for ein in range(40):
        #    self.valoptimizer.zero_grad()
        #    _, values = self.acmodel.forward(preprocess_obss(exps['obs'], device=self.device))
        #    valueloss = ((all_Gs - values)**2 ).mean()
        #    valueloss.backward(retain_graph=True)

        #    self.valoptimizer.step()

        print('valuepart ', valuepart.item())


class a2c(BaseAlgo):
    """standard actor critic algorithm"""

    def __init__(self, envs, acmodel, device=None, args=None, preprocess_obss=None, mode='train'):

        super().__init__(envs, acmodel, device, args=args, preprocess_obss=preprocess_obss, mode=mode)

        if args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.acmodel.parameters(), args.lr)
            self.valoptimizer = torch.optim.SGD(self.acmodel.critic.parameters(), args.lr)
        elif args.optimizer == 'Adam': 
            self.optimizer = torch.optim.Adam(self.acmodel.parameters(), args.lr)
            #self.valoptimizer = torch.optim.Adam(self.acmodel.critic.parameters(), args.lr)
        else:
            raise ValueError('Wrong optimizer name')


    def update_parameters(self, exps, preprocess_obss=None):

        T = exps['rewards'].shape[0]
        #_, values = self.acmodel.forward(preprocess_obss(exps['obs'], device=self.device))

        #nextvalues = torch.cat([values[1:], torch.zeros(1).to(self.device)], dim=0)

        all_Gs = torch.zeros(T, self.num_procs).to(self.device)
        for t in range(T):
            gams = ((torch.ones(T - t, self.num_procs) * self.discount) ** (torch.arange(T-t).unsqueeze(1).repeat(1, self.num_procs)).float()).to(self.device)
            #all_Gs.append((exps['rewards'][t:T] * gams).sum(0, keepdim=True).to(self.device))
            all_Gs[t] = (exps['rewards'][t:T] * gams).sum(0).to(self.device)

        #all_Gs = torch.tensor(all_Gs).to(self.device) 

        #for nein in range(1):
        #    self.valoptimizer.zero_grad()
            #_, values = self.acmodel.forward(preprocess_obss(exps['obs'], device=self.device))
        value_loss = 0.5*(all_Gs  - exps['all_values']).pow(2).mean()
        #    valuepart.backward(retain_graph=True)
        #    self.valoptimizer.step()
        #print(valuepart.item())


        gae_lambda = 0.95
        with torch.no_grad():
            advantages = torch.zeros(self.num_frames_per_proc, self.num_procs).to(self.device)
            for i in reversed(range(self.num_frames_per_proc)):
                #next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
                next_value = exps['all_values'][i+1] if i < self.num_frames_per_proc - 1 else exps['all_values'][-1]
                next_advantage = advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

                delta = exps['rewards'][i] + self.discount * next_value - exps['all_values'][i]
                advantages[i] = delta + self.discount * gae_lambda * next_advantage #* next_mask
        advantages.detach()


        #with torch.no_grad():
        #    _, values = self.acmodel.forward(preprocess_obss(exps['obs'], device=self.device))

        #    nextvalues = torch.cat([values[1:], torch.zeros(1).to(self.device)], dim=0)

        #    deltas = exps['rewards'] + self.discount*nextvalues - values
        #    #deltas = deltas * gams


        self.optimizer.zero_grad()
        log_p = (- (advantages*exps['log_probs'][:T]).mean())  
        loss = log_p + value_loss - exps['all_entropy'].mean()
        print(log_p.item())
        log_p.backward(retain_graph=True)

        self.optimizer.step()


class GModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False, algo='reinforce'):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.algo = algo

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        # Resize image embedding
        self.embedding_size = self.image_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        if self.algo in ['a2c', 'reinforce_wbase']:
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)


    def forward(self, obs):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        #print(x)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        embedding = x

        x = self.actor(embedding)
        #print(F.log_softmax(x,dim=1))
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        if self.algo in ['a2c', 'reinforce_wbase']:
            x = self.critic(embedding)
            value = x.squeeze(1)
        else:
            value = None

        return dist, value


