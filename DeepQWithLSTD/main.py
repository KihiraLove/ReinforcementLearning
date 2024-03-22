from collections import namedtuple, deque
from itertools import count
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import math
import random
import torch
import time

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
ALPHA = 0.001

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))


# Define Deep Q-Network (DQN) with autoencoder for MountainCar
class AutoencoderDQN(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=128):
        super(AutoencoderDQN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space)
        )
        self.decoder = nn.Sequential(
            nn.Linear(action_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_space)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class LSTD:
    def __init__(self, input_dim, output_dim, gamma, learning_rate):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.A = np.eye(input_dim)
        self.b = np.zeros((input_dim, 1))

    def update(self, state, next_state, reward):
        phi = state.reshape(-1, 1)
        next_phi = next_state.reshape(-1, 1)
        target = reward + self.gamma * np.max(self.q(next_phi))

        self.A += np.dot(phi, (phi - self.gamma * next_phi).T)
        self.b += phi * target

    def q(self, state):
        return np.dot(self.output_weights(), state)

    def output_weights(self):
        return np.dot(np.linalg.inv(self.A), self.b)


class ExperienceBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append(Experience(state, action, next_state, reward))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def select_action(env, state, steps_done, policy_model):
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_model(state).max(1).indices.view(1, 1), steps_done
    else:
        return torch.tensor([[env.action_space.sample()]], device='cpu', dtype=torch.long), steps_done


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)


def optimize_model():
    if experience_buffer.__len__() < BATCH_SIZE:
        return
    transitions = experience_buffer.sample(BATCH_SIZE)
    batch = Experience(*zip(*transitions))
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_model(state_batch)
    state_action_values = state_action_values.gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    loss = loss_function(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_model.parameters(), 100)
    optimizer.step()


mountain_car_env_name = 'MountainCar-v0'
acrobot_env_name = 'Acrobot-v1'
pendulum_env_name = 'Pendulum-v0'
episodes = 500
max_steps = 500
episode_grouping = 10

steps_done = 0
episode_durations = []

plt.ion()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(mountain_car_env_name)
state, _ = env.reset()
n_observations = len(state)
n_actions = env.action_space.n

policy_model = AutoencoderDQN(n_observations, n_actions).to(device)
target_model = AutoencoderDQN(n_observations, n_actions).to(device)
target_model.load_state_dict(policy_model.state_dict())

loss_function = nn.MSELoss()
optimizer = optim.AdamW(policy_model.parameters(), lr=ALPHA, amsgrad=True)

# lstd = LSTD(n_observations, n_actions, GAMMA, ALPHA)
experience_buffer = ExperienceBuffer(capacity=5000)

# Train autoencoder
while episodes > 0:
    for episode in range(episode_grouping):
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        terminated = False
        episode_duration = 0
        while not terminated and episode_duration < max_steps:
            action, steps_done = select_action(env, state, steps_done, policy_model)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            experience_buffer.add(state, action, reward, next_state)
            state = next_state
            optimize_model()

            target_net_state_dict = target_model.state_dict()
            policy_net_state_dict = policy_model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            target_model.load_state_dict(target_net_state_dict)
            episode_duration += 1
            if done or episode_duration == 500:
                episode_durations.append(episode_duration)
                plot_durations(episode_durations)

        episodes -= 1

print('Complete')
plot_durations(episode_durations, show_result=True)
plt.ioff()
plt.show()
