import torch 
import routines as rt
import utils 
import argparse
import numpy as np
from array2gif import write_gif

parser = argparse.ArgumentParser()


parser.add_argument("--algo", required=True,
                    help="algorithm to use: a2c | ppo (REQUIRED)")
parser.add_argument("--env", required=True,
                    help="name of the environment to train on (REQUIRED)")
parser.add_argument("--model", default=None,
                    help="name of the model (default: {ENV}_{ALGO}_{TIME})")
parser.add_argument("--seed", type=int, default=1,
                    help="random seed (default: 1)")
parser.add_argument("--render", type=int, default=1, help='rendering the interactions')
parser.add_argument("--render_pause", type=float, default=0.1, help='rendering the interactions')
parser.add_argument("--frames-per-proc", type=int, default=100,
                    help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--optimizer", type=str, default='SGD', help='the optimizer in [SGD, Adam]')

parser.add_argument("--num_episodes", type=int, default=500)

args = parser.parse_args()

env = utils.make_env(args.env, args.seed)
envs = [env]

obs_space, preprocess_obss = utils.get_obss_preprocessor(env.observation_space)
acmodel = rt.GModel(obs_space, env.action_space, algo=args.algo)

#modelpath = 'algo_results/MiniGrid-Empty-5x5-v0_policy_gradient_lr0.001_gam0.99_seed1_19-12-13-18-48-58.t'
#modelpath = 'algo_results/MiniGrid-Empty-5x5-v0_policy_gradient_lr0.001_gam0.99_seed1_19-12-16-15-58-28.t'
#modelpath = 'algo_results/MiniGrid-Empty-5x5-v0_policy_gradient_lr0.001_gam0.99_seed1_19-12-16-16-04-32.t'
#modelpath = 'algo_results/MiniGrid-Empty-5x5-v0_policy_gradient_lr0.0004_seed1_19-12-16-18-24-55.t'
#modelpath = 'algo_results/MiniGrid-Empty-5x5-v0_reinforce_wbase_lr0.0003_Adam_seed1_19-12-18-16-12-32.t'
#modelpath = 'algo_results/MiniGrid-Empty-5x5-v0_a2c_lr0.0003_Adam_seed1_19-12-28-16-34-46.t'
modelpath = 'algo_results/MiniGrid-Empty-5x5-v0_a2c_lr0.0003_Adam_seed1_19-12-28-17-46-12.t'

# start plotting things

params = torch.load(modelpath)

acmodel.load_state_dict(params) 

# load the algo to generate experiences (the algo we choose doesn't matter) 
algo = rt.a2c(envs, acmodel=acmodel, device='cpu', args=args, preprocess_obss=preprocess_obss, mode='test') 
                          

# the function below collects the experiences and renders the interactions 
exps = algo.collect_experiences_basic()
print('all rewards {}'.format(exps['rewards']))
print('total reward {}'.format(exps['rewards'].sum()))

print("Saving gif... ", end="")
write_gif(np.array(exps['all_frames']), modelpath+".gif", fps=1/args.render_pause)
print("Done.")



