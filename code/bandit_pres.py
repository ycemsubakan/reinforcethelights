import torch
import matplotlib.pyplot as plt
import pdb
import numpy as np

torch.manual_seed(0)

K = 4
means = torch.randn(K)
all_rewards = []
N = 100

X = means.unsqueeze(0) + torch.randn(N, K)
ax = plt.violinplot(X.numpy())
plt.xticks(np.arange(1,5))

fs = 18 
plt.xlabel('$A_t$', fontsize=fs)
plt.ylabel('$R_t$', fontsize=fs)

plt.savefig('../pres/bandit_distr.png')
## do the gui thing

T = 5
for t in range(5):
    a = int(input('Choose your action between 1 .. {}'.format(K))) - 1
    reward = means[a].item() + torch.randn(1).item() 

    all_rewards.append(reward)
    print('your reward is {}'.format(reward))

print('your total reward {}'.format(sum(all_rewards)))


