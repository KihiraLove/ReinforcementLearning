import time

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import collections


class ArgmaxActionSelector:
    """
    Selects actions using argmax
    """
    def __call__(self, scores):
        assert isinstance(scores, np.ndarray)
        return np.argmax(scores, axis=1)


class LSDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(LSDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, n_actions)
        # self.fc = nn.Sequential(
        #     nn.Linear(conv_out_size, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, n_actions)
        # )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x, features=False):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc2(self.relu1(self.fc1(conv_out)))

    def forward_to_last_hidden(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.relu1(self.fc1(conv_out))


def preprocessor(states):
    """
    Convert list of states into the form suitable for model. By default, we assume Variable
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


class DQNAgent:
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__(self, dqn_model, action_selector, device=torch.device("cpu")):
        self.dqn_model = dqn_model
        self.action_selector = action_selector
        self.device = device

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)
        states = preprocessor(states)
        if torch.is_tensor(states):
            states = states.to(self.device)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = self.action_selector(q)
        return actions, agent_states


env = gym.make('MountainCar-v0')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(env.observation_space.shape)
print(env.observation_space.shape[0])
net = LSDQN(env.observation_space.shape, env.action_space.n).to(device)
selector = ArgmaxActionSelector()
agent = DQNAgent(net, selector, device=device)
state = env.reset()
total_reward = 0.0
c = collections.Counter()
while True:
    start_ts = time.time()
    env.render()
    # state_v = torch.tensor(np.array([state], copy=False))
    # state_v = ptan.agent.default_states_preprocessor(state)
    # q_vals = net(state_v).data.numpy()[0]
    # action = np.argmax(q_vals)
    action, _ = agent([state])
    # print(action)
    c[action[0]] += 1
    state, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated:
        env.close()
        break

    delta = 1 / 25 - (time.time() - start_ts)
    if delta > 0:
        time.sleep(delta)
print("Total reward: %.2f" % total_reward)
print("Action counts:", c)