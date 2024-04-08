import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple, deque
import copy
import numpy as np

from DQNAgent import DQNAgent
from ExperienceBuffer import ExperienceBuffer
from ExperienceSource import ExperienceSource
from Functions import decay_epsilon, unpack_batch, ls_step, chose_random_action
from LSDQN import LSDQN
from RewardTracker import RewardTracker
from TargetModel import TargetModel

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.05
DECAY_EVERY = 1000
TAU = 0.005
ALPHA = 0.001
MAX_EPISODES = 1000
BUFFER_SIZE = 5000
MAX_STEPS = 500
EPISODE_GROUPING = 10
n_drl = 5000
LAMBDA = 1
END_REWARD = 300
epsilon = 0.1
n_srl = BATCH_SIZE  # size of batch in SRL step
target_update_freq = 1000
terminated = False


ExperienceFirstLast = namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make("Acrobot-v1")
lsdqn_model = LSDQN(env.observation_space.shape, env.action_space.n).to(device)
target_model = TargetModel(lsdqn_model)
agent = DQNAgent(lsdqn_model, device)

experience_source = ExperienceSource(env, agent, GAMMA)
experience_source_iter = iter(experience_source)
experience_buffer = ExperienceBuffer(BUFFER_SIZE)
optimizer = optim.Adam(lsdqn_model.parameters(), lr=ALPHA)

reward_tracker = RewardTracker()
drl_updates = 0

while len(experience_buffer) < BUFFER_SIZE:
    state = env.reset()
    done = False
    while not done:
        action = chose_random_action(env.action_space)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        experience_buffer.add(Experience(state, action, reward, next_state, terminated))
        state = next_state
        if len(experience_buffer) >= BUFFER_SIZE:
            break


for episode in range(MAX_EPISODES):
    state = env.reset()
    done = False

    while not done:
        new_rewards = experience_source.pop_total_rewards()
        if new_rewards:
            if terminated:
                break

        optimizer.zero_grad()
        batch = experience_buffer.get_batch(BATCH_SIZE)

        # Calculate Loss
        states, actions, rewards, dones, next_states = unpack_batch(batch)

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = lsdqn_model(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        next_state_values = target_model.target_model(next_states_v).max(1)[0]
        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * GAMMA + rewards_v
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)

        loss_v.backward()
        optimizer.step()
        drl_updates += 1

        # LS-UPDATE STEP
        if (drl_updates % n_drl == 0) and (len(experience_buffer) >= n_srl):
            print("performing ls step...")
            batch = experience_buffer.get_batch(n_srl)
            ls_step(lsdqn_model, target_model.target_model, batch, GAMMA, len(batch), LAMBDA, BATCH_SIZE, device)

        if episode % target_update_freq == 0 and episode != 0:
            target_model.sync()

    epsilon = decay_epsilon(epsilon)
