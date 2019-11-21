import gym
import numpy as np
# import pdb


class RandomAgent():

    def __init__(self, obs_space, action_space):
        self.action_space = action_space

    def get_action(self, state):
        return self.action_space.sample()


env = gym.make('Blackjack-v0')
print(env.action_space)
print(env.observation_space)
agent = RandomAgent(env.observation_space, env.action_space)

N_EP = 100000
R_list = []
n_win = 0
n_draw = 0
n_lost = 0

for i_episode in range(N_EP):
    s_t = env.reset()
    R = 0.
    for t in range(100):
        a_t = agent.get_action(s_t)
        s_tp1, r_t, done, info = env.step(a_t)
        # print(s_t, a_t, r_t, s_tp1)
        s_t = s_tp1
        R += r_t
        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            break
    # print('Return: {}'.format(R))
    if R < 0:
        n_lost += 1
    elif R > 0:
        n_win += 1
    else:
        n_draw += 1
    R_list.append(R)
env.close()
R_arr = np.array(R_list)
print('----------Statistics-------------')
print('Avg. return: {} ({})'.format(np.mean(R_arr), np.std(R_arr)))
print('%win/draw/lost: {}/{}/{}'.format(n_win/N_EP, n_draw/N_EP, n_lost/N_EP))
