import torch.nn as nn


class LSDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(LSDQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.last_layer = nn.Linear(64, n_actions)

    def forward(self, x):
        return self.last_layer(self.network(x))

    def forward_to_last_hidden(self, x):
        return self.network(x)
