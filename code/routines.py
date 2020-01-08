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
import random
import math
from collections import namedtuple

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
        self.action_space_n = action_space.n

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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(object): 

    def __init__(self, target_net, policy_net, device, preprocess_obss):

        self.batch_size = 128
        self.gamma = 0.999
        self.eps_start = 0.9
        self.eps_end = 0.30
        self.eps_decay = 3000
        self.target_update = 10

        self.memory = ReplayMemory(10000)
        self.target_net = target_net
        self.policy_net = policy_net
        self.steps_done = 0 
        self.device = device
        self.preprocess_obss = preprocess_obss

        self.optimizer = torch.optim.Adam(policy_net.parameters())

        #self.optimizer = optim

    def optimize_model(self):

        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device) #, dtype=torch.uint8)
        non_final_next_states = self.preprocess_obss(batch.next_state, device=self.device)

        #non_final_next_states = torch.cat([self.preprocess_obss([s], device=self.device).image for s in batch.next_state if s is not None])
        state_batch = self.preprocess_obss(batch.state, device=self.device)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch)[0].logits.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states)[0].logits.max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def select_action(self, state):
        if 1: 
            n_actions = self.policy_net.action_space_n
            sample = random.random()
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
            self.eps_threshold = eps_threshold

            if self.steps_done > 10000:
                self.eps_threshold = 0.1
            #eps_threshold = 0.5
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.max(1) will return largest column value of each row.
                    # second column on max result is index of where max element was
                    # found, so we pick action with the larger expected reward.
                    dist, _ = self.policy_net(state)
                    return dist.logits.max(1)[1].view(1, 1)
            else:
                return torch.tensor([[random.randrange(n_actions)]], device=self.device, dtype=torch.long)
        else:
            self.eps_threshold = 0
            dist, _ = self.policy_net(state)
            return dist.sample().view(1, 1)  #logits.max(1)[1].view(1, 1)



