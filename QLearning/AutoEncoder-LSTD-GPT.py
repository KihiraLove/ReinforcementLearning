import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


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
        reconstructed = self.decoder(latent_code)
        return self.output_layer(latent_code), reconstructed


# Define replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state):
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return states, actions, rewards, next_states


class LSTD:
    def __init__(self, input_dim, output_dim, gamma, learning_rate):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.A = np.eye(input_dim)
        self.b = np.zeros((input_dim, 1))

    def update(self, state, action, reward, next_state):
        phi = state.reshape(-1, 1)
        next_phi = next_state.reshape(-1, 1)
        target = reward + self.gamma * np.max(self.Q(next_phi))

        self.A += np.dot(phi, (phi - self.gamma * next_phi).T)
        self.b += phi * target

    def Q(self, state):
        return np.dot(self.output_weights(), state)

    def output_weights(self):
        return np.dot(np.linalg.inv(self.A), self.b)


# Define autoencoder training function
def train_autoencoder(env, autoencoder, replay_buffer, num_episodes=10):
    input_dim = env.observation_space.shape[0]
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state)
            state = next_state

        # Retrieve states from replay buffer
        states, _, _, _ = zip(*replay_buffer.buffer)
        if len(states) == 0:
            print("Replay buffer is empty. Skipping training for this episode.")
            continue  # Skip training if replay buffer is empty

        # Convert states to PyTorch tensor
        states_tensor = torch.tensor(states, dtype=torch.float32)
        print("States tensor shape:", states_tensor.shape)

        optimizer.zero_grad()
        _, reconstructed = autoencoder(states_tensor)
        loss = criterion(reconstructed, states_tensor)
        loss.backward()
        optimizer.step()
        print(f"Autoencoder Episode [{episode + 1}/{num_episodes}], Loss: {loss.item()}")


# Define LSTD training function
def train_lstd(env, autoencoder, replay_buffer, num_episodes=10):
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    lstd = LSTD(input_dim, output_dim, gamma=0.99, learning_rate=0.001)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state)
            state = next_state

        states, _, _, _ = zip(*replay_buffer.buffer)
        states = torch.tensor(states, dtype=torch.float32)
        latent_codes, _ = autoencoder.encoder(states)
        latent_codes = latent_codes.detach().numpy()

        for i in range(len(replay_buffer.buffer)):
            state, action, reward, next_state = replay_buffer.buffer[i]
            lstd.update(latent_codes[i], action, reward,
                        latent_codes[i + 1] if i + 1 < len(replay_buffer.buffer) else latent_codes[i])

        print(f"LSTD Episode [{episode + 1}/{num_episodes}]")

        # Update autoencoder last hidden layer
        last_hidden_weights = lstd.output_weights()
        autoencoder.decoder[0].weight.data = torch.tensor(last_hidden_weights.T, dtype=torch.float32)


# Main training loop
env = gym.make("CartPole-v1")
num_episodes = 25000
max_steps = 500
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
replay_buffer = ReplayBuffer(capacity=10000)

autoencoder_dqn = AutoencoderDQN(input_dim, output_dim)
train_autoencoder(env, autoencoder_dqn, replay_buffer, num_episodes=10)
train_lstd(env, autoencoder_dqn, replay_buffer, num_episodes=10)

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
        with torch.no_grad():
            print(state)
            state_tensor = torch.tensor(state, dtype=torch.float32)
            q_values, _ = autoencoder_dqn(state_tensor)
            action = torch.argmax(q_values).item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        replay_buffer.add(state, action, reward, next_state)
        state = next_state
        if done:
            break
    print(f"Episode [{episode + 1}/{num_episodes}], Total Reward: {total_reward}")

