import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Define a simple neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)

# Initialize MountainCar environment
env = gym.make("MountainCar-v0", render_mode="human")
input_size = env.observation_space.shape[0]  # State space dimension
output_size = env.action_space.n  # Number of actions
model = NeuralNetwork(input_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_episodes = 1000
max_steps = 200
rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):

        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Forward pass through the model to get action probabilities
        action_probs = torch.softmax(model(state_tensor), dim=1)

        # Sample action from the action probabilities
        action = np.random.choice(output_size, p=action_probs.detach().numpy().flatten())

        # Take action in the environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        # Update episode reward
        episode_reward += reward

        # Update the policy network
        optimizer.zero_grad()
        loss = criterion(action_probs, torch.tensor([action]))
        loss.backward()
        optimizer.step()

        # Transition to next state
        state = next_state

        if done:
            break

    rewards.append(episode_reward)

    # Print episode information
    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward}")

# Close the environment
env.close()

# Plot rewards
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Reward per Episode")
plt.show()
