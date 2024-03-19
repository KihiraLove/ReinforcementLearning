import gymnasium as gym
import argparse
import torch
import torch.optim as optim
import collections
import time
import os
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make("MountainCar-v0")
