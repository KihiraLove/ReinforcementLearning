import torch
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym
import numpy as np
import math
import random

import matplotlib.pyplot as plt
import matplotlib
from collections import deque, namedtuple
from itertools import count
from PIL import Image
import itertools


class Network(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))

    def forward(self, t):
        t = self.net(t)
        return t

    def act(self, state):
        pass


def decay(eps_start, eps_end, eps_decay_rate, current_step):
    return eps_end + (eps_start - eps_end) * np.exp(-1, eps_decay_rate * current_step)


env = gym.make('CartPole-v1')
env.reset()
