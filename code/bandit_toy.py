import torch
import pdb
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

def full_history_smoother(means, T=1000, K=10):
    rs = []
    Q = torch.zeros(K)
    all_actions = [] 
    all_rewards = [] 
    for t in range(T):
        if t < K:
            a = t
        else:
            a = torch.argmax(Q).item()
        
        reward = means[a].item() + torch.randn(1).item()
        all_actions.append(a)
        all_rewards.append(reward)

        A = torch.Tensor(all_actions)
        R = torch.Tensor(all_rewards)

        if t >= K:
            for k in range(K):
                inds = (A == k)
                Q[k] = R[inds].mean()  

    return R

def sequential_smoother(means, T=1000, K=10, epsilon=0):
    rs = []
    Q = torch.zeros(K)
    N = torch.zeros(K)
    all_actions = [] 
    all_rewards = [] 
    
    a = 0
    
    for t in range(T):

        #if t < K:
        #    a = t
        #else:
        if torch.rand(1).item() < epsilon:
            a = torch.randint(K, (1, )) 
        else:
            a = torch.argmax(Q).item()

        reward = means[a].item() + torch.randn(1).item()
        N[a] = N[a] + 1
        Q[a] = Q[a] + (1/N[a])*(reward - Q[a])
                
        all_rewards.append(reward)

    R = torch.Tensor(all_rewards)

    return R

K = 10
means = torch.randn(K) 

T = 300
epsilon = 0.1

R_fhist = full_history_smoother(means=means, T=T, K=K)
R_seq_ep0 = sequential_smoother(means=means, T=T, K=K, epsilon=0)
R_seq_epnz = sequential_smoother(means=means, T=T, K=K, epsilon=epsilon)


print('action average rewards {}'.format(means))
plt.subplot(311)
plt.plot(R_seq_ep0, label='sequential_ep0')
plt.plot(R_seq_epnz, label='sequential_ep{}'.format(epsilon))

plt.plot(R_fhist, label='full_history')

plt.xticks([])

plt.subplot(312)
plt.plot(torch.cumsum(R_seq_ep0, dim=0)/torch.arange(T), label='sequential_ep0')
plt.plot(torch.cumsum(R_seq_epnz, dim=0)/torch.arange(T), label='sequential_ep{}'.format(epsilon))

plt.plot(torch.cumsum(R_fhist, dim=0)/torch.arange(T), label='full_history')

plt.legend()

plt.subplot(313)
data = means.unsqueeze(0) + torch.randn(100, K)

ax = plt.violinplot(data.numpy())
plt.xticks(np.arange(1, K))

fs = 18 
plt.xlabel('$A_t$', fontsize=fs)
plt.ylabel('$R_t$', fontsize=fs)

plt.savefig('../pres/simple_bandit_algos.png')





