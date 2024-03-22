import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym


class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()  # Sigmoid activation for bounded output values
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


num_epochs = 500
num_samples = 500
# Initialize environment
env = gym.make('MountainCar-v0')
input_size = env.observation_space.shape[0]  # Size of input state
hidden_size = 64  # Size of hidden layer

# Initialize autoencoder
autoencoder = Autoencoder(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(num_samples):
        # Sample a state from the environment
        state, _ = env.reset()
        state_tensor = torch.tensor(state, dtype=torch.float32)

        # Forward pass
        reconstructed_state = autoencoder(state_tensor)

        # Compute loss
        loss = criterion(reconstructed_state, state_tensor)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print average loss for the epoch
    avg_loss = total_loss / num_samples
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
