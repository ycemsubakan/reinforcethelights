import torch
import pdb
import copy


def policy_evaluation(
        pi,
        Prsa,
        Pspsa,
        possible_r,
        gamma=0.5,
        threshold=1e-6,
        max_iter=100000):
    V = torch.rand(4)
    delta = 1.
    i = 0
    while delta > threshold and i < max_iter:
        Prspi = Prsa.gather(2, pi.repeat(
            Prsa.size(0), 1).unsqueeze(-1).long())
        reward_term = (Prspi * possible_r).sum(0)

        Pspspi = Pspsa.gather(2, pi.repeat(
            Pspsa.size(0), 1).unsqueeze(-1).long())
        state_term = gamma * (V.view(-1, 1, 1) * Pspspi).sum(0)

        V2 = (reward_term + state_term).max(-1)[0]
        delta = (V2-V).sum().abs()
        V = V2
        i += 1
    print('Policy evaluation converged after {} epoch'.format(i))
    return V


def policy_improvement(
        Prsa,
        Pspsa,
        possible_r,
        gamma=0.5,
        threshold=1e-6,
        max_iter=100000):

    pi = torch.randint(0, 2, (4, )).byte()
    delta = 1.
    i = 0
    while delta > threshold and i < max_iter:
        V = policy_evaluation(pi, Prsa, Pspsa, possible_r)
        reward_term = (Prsa * possible_r).sum(0)
        state_term = gamma * (V.view(-1, 1, 1) * Pspsa).sum(0)
        pi2 = (reward_term + state_term).max(-1)[1]
        delta = (pi-pi2).sum().abs()
        pi = pi2
        i += 1
    print('Policy improvement converged after {} epoch'.format(i))
    return pi


def value_iteration(
        Prsa,
        Pspsa,
        possible_r,
        gamma=0.5,
        threshold=1e-6,
        max_iter=100000):

    V = torch.rand(4)
    delta = 1.
    i = 0
    while delta > threshold and i < max_iter:
        reward_term = (Prsa * possible_r).sum(0)
        state_term = gamma * (V.view(-1, 1, 1) * Pspsa).sum(0)

        V2 = (reward_term + state_term).max(-1)[0]
        delta = (V2-V).sum().abs()
        V = V2
        i += 1
    pi = (reward_term + state_term).argmax(-1)
    print('Value iteration converged after {} epoch'.format(i))
    return pi, reward_term, state_term


def simulate(x1, x2, pi):
    twait = 0
    all_actions = []
    while (len(x1) > 0) or (len(x2) > 0):

        # Create the state
        if len(x1) > 0 and len(x2) > 0:
            state = (x1[0], x2[0])
        elif len(x1) == 0 and len(x2) > 0:
            state = (0, x2[0])
            x1 = [0]
        elif len(x1) > 0 and len(x2) == 0:
            state = (x1[0], 0)
            x2 = [0]

        # Retrieve the action and modify the state
        a = pi.get_action(state)
        all_actions.append(a)
        if a == 0:
            x1.pop(0)
            if x2[0] == 0:
                x2.pop(0)
        else:
            x2.pop(0)
            if x1[0] == 0:
                x1.pop(0)

        twait = twait + 1

    # print(all_actions)
    return twait


class Policy():

    def __init__(self, pi, possible_S):
        self.pi = pi
        self.possible_S = possible_S

    def get_action(self, state):
        state_index = self.possible_S.index(state)
        return self.pi[state_index]

    def update(self, pi):
        self.pi = pi


class HeuristicPolicy():

    def __init__(self):
        self.action = 0

    def get_action(self, state):
        # pdb.set_trace()
        if state[0] == 1:
            self.action = 0
        elif state[1] == 1:
            self.action = 1
        return self.action


def build_markov_model():
    Prsa = torch.zeros(2, 4, 2)
    Pspsa = torch.zeros(4, 4, 2)

    possible_S = [(0, 0), (1, 0), (0, 1), (1, 1)]
    possible_r = torch.tensor([0, 1]).view(-1, 1, 1).float()
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
                    Pspsa[:, s, a] = torch.tensor(
                        [(1-p1)*(1-p2), p1*(1-p2), (1-p1)*p2, p1*p2])
                else:
                    Pspsa[:, s, a] = torch.tensor([0, 0, 1-p1, p1])
            else:
                if s in [0, 2]:
                    Pspsa[:, s, a] = torch.tensor(
                        [(1-p1)*(1-p2), p1*(1-p2), (1-p1)*p2, p1*p2])
                else:
                    Pspsa[:, s, a] = torch.tensor([0, 1-p2, 0, p2])
    return Prsa, Pspsa, possible_r, possible_S


if __name__ == '__main__':

    torch.manual_seed(6666)

    N = 10
    p1 = 0.5
    p2 = 0.5
    x1 = [torch.randint(0, 2, (1, )).byte().item() for n in range(N)]
    x2 = [torch.randint(0, 2, (1, )).byte().item() for n in range(N)]

    Prsa, Pspsa, possible_r, possible_S = build_markov_model()

    hpolicy = HeuristicPolicy()
    twait_h = simulate(copy.deepcopy(x1), copy.deepcopy(x2), hpolicy)

    policy_improvement(Prsa, Pspsa, possible_r)
    pi, reward_term, state_term = value_iteration(Prsa, Pspsa, possible_r)

    rlpolicy = Policy(pi, possible_S)
    twait_rl = simulate(copy.deepcopy(x1), copy.deepcopy(x2), rlpolicy)
    print('total wait time heuristic {}'.format(twait_h))
    print('total wait time reinforcement {}'.format(twait_rl))
    # pdb.set_trace()
