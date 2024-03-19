import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define a simple autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()  # Sigmoid activation for reconstruction
        )

    def forward(self, x):
        latent_code = self.encoder(x)
        reconstructed = self.decoder(latent_code)
        return reconstructed


# Preprocess state observations
def preprocess_state(state):
    return np.array(state).flatten()


# Train the autoencoder
def train_autoencoder(env_name, latent_dim, num_epochs, lr):
    # Create environment
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]

    # Initialize autoencoder
    autoencoder = Autoencoder(input_dim, latent_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    # Train the autoencoder
    for epoch in range(num_epochs):
        total_loss = 0
        for _ in range(env.spec.max_episode_steps):  # Limit training to one episode per epoch
            state = env.reset()
            state = preprocess_state(state)
            state = torch.tensor(state, dtype=torch.float32)

            # Forward pass
            optimizer.zero_grad()
            reconstructed = autoencoder(state)
            loss = criterion(reconstructed, state)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / env.spec.max_episode_steps:.4f}")

    return autoencoder


# Example usage
env_names = ['Acrobot-v1', 'Pendulum-v0', 'MountainCar-v0']
latent_dim = 16
num_epochs = 50
lr = 0.001

for env_name in env_names:
    print(f"Training autoencoder for environment: {env_name}")
    autoencoder = train_autoencoder(env_name, latent_dim, num_epochs, lr)
