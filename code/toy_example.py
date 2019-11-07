import torch 
import visdom
import pdb
import copy

N = 10
p1 = 0.5
p2 = 0.5
x1 = [torch.randint(0, 2, (1, )).byte().item() for n in range(N)] 
x2 = [torch.randint(0, 2, (1, )).byte().item() for n in range(N)]

x1org = copy.deepcopy(x1)
x2org = copy.deepcopy(x2)

# solve the problem with handcrafted policy
y = []
twait = 0
all_actions = []
a = 1
print('total vehicles {}'.format(sum(x1) + sum(x2)))
while (len(x1) > 0) or (len(x2) > 0): 
    print('first list {}'.format(x1))
    print('second list {}'.format(x2))

    if len(x1) > 0 and len(x2) > 0: 
        s = (x1[0], x2[0])
    elif len(x1) == 0 and len(x2) > 0:
        s = (0, x2[0])
        x1 = [0]

    if s[0] == 1: 
        x1.pop(0)
        if s[1] == 0:
            x2.pop(0)
        a = 1
    elif s[1] == 1:
        x2.pop(0)
        if s[0] == 0:
            x1.pop(0)
        a = 2
    else:
        x1.pop(0)
        x2.pop(0)
    all_actions.append(a)
    twait = twait + 1

print(all_actions)
print('total wait time {}'.format(twait))


## now, try to learn the thing with value iteration 
Prsa = torch.zeros(2, 4, 2)
Pspsa = torch.zeros(4, 4, 2)

possible_S = [(0, 0), (1, 0), (0, 1), (1, 1)]
possible_r = torch.tensor([0, 1]).float()

for s in range(4):
    for a in range(2):
        # fill in the reward table
        if possible_S[s][a] == 1:
            Prsa[:, s, a] = torch.tensor([0, 1]) 
        else:
            Prsa[:, s, a] = torch.tensor([1, 0])

        # fill in the next state table
        if a == 0: 
            if s < 2: 
                Pspsa[:, s, a] = torch.tensor([(1-p1)*(1-p2), p1*(1-p2), (1-p1)*p2, p1*p2])
            else: 
                Pspsa[:, s, a] = torch.tensor([0, 0, 1-p1, p1])
        else:
            if s in [0, 2]:
                Pspsa[:, s, a] = torch.tensor([(1-p1)*(1-p2), p1*(1-p2), (1-p1)*p2, p1*p2])
            else:
                Pspsa[:, s, a] = torch.tensor([0, 1-p2, 0, p2])

V = torch.rand(4)
gam = 0.5

# do the value iterations
for e in range(100):
    reward_term = (Prsa * possible_r.unsqueeze(-1).unsqueeze(-1)).sum(0) 
    state_term = gam * (V.unsqueeze(-1).unsqueeze(-1) * Pspsa).sum(0)

    V = (reward_term + state_term).max(-1)[0]
    print(V)

pi = (reward_term + state_term).argmax(-1)

twait_rl = 0
all_actions_rl = []
## see how well the policy above does
while (len(x1org) > 0) or (len(x2org) > 0): 
    print('first list {}'.format(x1org))
    print('second list {}'.format(x2org))

    if len(x1org) > 0 and len(x2org) > 0: 
        state = (x1org[0], x2org[0])
    elif len(x1org) == 0 and len(x2org) > 0:
        state = (0, x2org[0])
        x1org = [0]
    elif len(x1org) >  0 and len(x2org) == 0:
        state = (x1org[0], 0)
        x2org = [0]
        
    s = possible_S.index(state)

    a = pi[s]             
    na = 1 - a
    all_actions_rl.append(a)

    if a == 0:
        x1org.pop(0)
        if x2org[0] == 0:
            x2org.pop(0)
    else:
        x2org.pop(0)
        if x1org[0] == 0:
            x1org.pop(0)

    twait_rl = twait_rl + 1
    
print(all_actions_rl)

print('total wait time heuristic {}'.format(twait))
print('total wait time reinforcement {}'.format(twait_rl))

pdb.set_trace()