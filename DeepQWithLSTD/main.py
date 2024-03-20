import math
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import gymnasium as gym

EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000


# Define Deep Q-Network (DQN) with autoencoder for MountainCar
class AutoencoderDQN(nn.Module):
    def __init__(self, observation_space, action_space, hidden_dim=128):
        super(AutoencoderDQN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space)
        )
        self.decoder = nn.Sequential(
            nn.Linear(action_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, observation_space),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Define the LSTD algorithm
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


# Define replay buffer for experience replay
class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return model(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device='cpu', dtype=torch.long)


# Example usage
env_name = 'MountainCar-v0'
episodes = 25000
max_steps = 500
episode_grouping = 10
batch_size = 32
gamma = 0.99
alpha = 0.001

env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

model = AutoencoderDQN(input_dim, output_dim)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=alpha)
lstd = LSTD(input_dim, output_dim, gamma, alpha)
experience_buffer = ExperienceBuffer(capacity=5000)

# Taking random actions to collect data
for episode in range(episode_grouping):
    state = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        action = env.action_space.sample()
        next_state, reward, terminated, _, _ = env.step(action)
        total_reward += reward
        experience_buffer.add(state, action, reward, next_state)
        state = next_state

print("Experience buffer filled with 10 episodes of random data")

# Train autoencoder
while episodes > 0:
    for episode in range(episode_grouping):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        losses = []
        terminated = False
        while not terminated:
            env.render()
            encoded, decoded = model(state)
            loss = loss_function(decoded, state)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss)
            state, _, terminated, _, _ = env.step(encoded)
            state = torch.tensor(state, dtype=torch.float32)
            time.sleep(0.0001)

        env.close()
        episodes -= 1
