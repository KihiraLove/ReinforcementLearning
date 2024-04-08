import torch
import numpy as np


epsilon = 0.1


def chose_action(actions):
    if np.random.rand() < epsilon:
        return np.random.choice(len(actions))
    else:
        return np.argmax(actions)


class DQNAgent:
    def __init__(self, dqn_model, device):
        self.dqn_model = dqn_model
        self.device = device

    def __call__(self, states):
        if len(states) == 1:
            np_states = states[0][0]
        else:
            np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
        states = torch.tensor(np_states)
        if torch.is_tensor(states):
            states = states.to(self.device)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = chose_action(q)
        return actions
