import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Autoencoder class remains the same
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Modified part: Prepare the Environment Data and Collect Rewards
env = gym.make('MountainCar-v0')
observations = []
episode_rewards = []

for episode in range(100):  # Run 100 episodes
    observation = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = env.action_space.sample()  # Take a random action
        observation, reward, done, _, _ = env.step(action)
        total_reward += reward
        observations.append(observation)
    episode_rewards.append(total_reward)

observations = np.array(observations)
observations_tensor = torch.Tensor(observations)

# Creating a DataLoader
dataset = TensorDataset(observations_tensor, observations_tensor)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop remains the same
model = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, dataloader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            inputs, targets = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

train(model, dataloader, criterion, optimizer)

# Plotting the rewards
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards per Episode')
plt.show()
