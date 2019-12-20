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
import os
import visdom
import pickle

home = os.path.expanduser('~')
sys.path.insert(0, home + "/rl-starter-files")


import utils
import pdb
import matplotlib.pyplot as plt

# Parse arguments
import routines as rt


vis = visdom.Visdom(port=5800, server='http://nmf.cs.illinois.edu', env='main', use_incoming_socket=False)
assert vis.check_connection()


parser = argparse.ArgumentParser()

## General parameters
parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=1,
                    help="number of updates between two logs (default: 1)")
parser.add_argument("--save-interval", type=int, default=10,
                    help="number of updates between two saves (default: 10, 0 means no saving)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                    help="number of frames of training (default: 1e7)")
parser.add_argument("--epochs", type=int, default=4,
                    help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                    help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=100,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--optimizer", type=str, default='SGD', help='the optimizer in [SGD, Adam]')
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")
parser.add_argument("--render", type= int, default=0, help='rendering the interactions')
parser.add_argument("--num_episodes", type=int, default=500)
parser.add_argument("--save_model", type=int, default=1, help='save or not save the model')

args = parser.parse_args()

# Set run dir
date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
default_model_name = "{}_{}_lr{}_{}_seed{}_{}".format(args.env, args.algo, args.lr, args.optimizer, args.seed, date)

model_name = default_model_name
model_dir = utils.get_model_dir(model_name)

utils.seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = utils.make_env(args.env, args.seed)
envs = [env]

obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
acmodel = rt.GModel(obs_space, env.action_space, algo=args.algo)

if args.algo == 'reinforce':
    algo = rt.reinforce(envs, acmodel, device, args=args, preprocess_obss=preprocess_obss)
elif args.algo == 'reinforce_wbase':
    algo = rt.reinforce_wbaseline(envs, acmodel, device, args=args, preprocess_obss=preprocess_obss)
elif args.algo == 'a2c':
    algo = rt.a2c(envs, acmodel, device, args=args, preprocess_obss=preprocess_obss)
else:
    raise ValueError("Incorrect algorithm name: {}".format(args.algo))


reward = 0
done = False

all_rewards = [] 
for i in range(args.num_episodes):
    print('episode {}'.format(i))

    update_start_time = time.time()
    exps = algo.collect_experiences_basic()

    algo.update_parameters(exps, preprocess_obss)
    all_rewards.append(exps['rewards'].sum().item())
    #print(exps.reward)
    running_mean = torch.tensor(all_rewards).cumsum(0)/torch.arange(i+1).float()

    if i % 25 == 0 and i > 0:
        vis.line(all_rewards, win='all_rewards')
        vis.line(running_mean, win='running_mean')
    
        if args.save_model:
            pickle.dump({'all_rewards' : all_rewards, 'running_mean' : running_mean.numpy(), 'args': args.__dict__}, 
                    open('algo_results/' + model_name + '.pk', 'wb'))

if args.save_model:
    torch.save(acmodel.cpu().state_dict(), 'algo_results/' + model_name + '.t')

pdb.set_trace()

