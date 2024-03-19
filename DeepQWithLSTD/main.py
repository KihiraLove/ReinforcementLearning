import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import namedtuple, deque
import gymnasium as gym


# Define Deep Q-Network (DQN) with autoencoder
class AutoencoderDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(AutoencoderDQN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        latent_code = self.encoder(x)
        return self.output_layer(latent_code), self.decoder(latent_code)


# Define the Deep Q-Network (DQN) with feature extractor
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(DQN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.output_layer(features)


# Define the LSTD algorithm
class LSTD:
    def __init__(self, input_dim, output_dim, gamma, learning_rate):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.A = np.eye(input_dim)
        self.b = np.zeros((input_dim, 1))

    def update(self, state, next_state, action, reward):
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
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        return zip(*random.sample(self.buffer, batch_size))


# Example usage
env_name = 'CartPole-v1'
num_episodes = 25000
max_steps = 500
episode_grouping = 10
batch_size = 32
gamma = 0.99
alpha = 0.001

env = gym.make(env_name)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

dqn = DQN(input_dim, output_dim)
optimizer = optim.Adam(dqn.parameters(), lr=alpha)

replay_buffer = ReplayBuffer(capacity=10000)
lstd = LSTD(input_dim, output_dim, gamma, alpha)

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    terminated = False
    while not terminated:
        epsilon = 0.1
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = dqn(state_tensor)
                action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        replay_buffer.add(state, action, reward, next_state)

        if len(replay_buffer.buffer) >= batch_size:
            states, actions, rewards, next_states = replay_buffer.sample(batch_size)
            for i in range(batch_size):
                lstd.update(states[i], next_states[i], actions[i], rewards[i])

        state = next_state

    if episode % 100 == 0:
        print(f"Episode: {episode}, Total Reward: {total_reward}")
