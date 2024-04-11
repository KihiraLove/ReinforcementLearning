import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple, deque
import copy
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state'])

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.05
DECAY_EVERY = 10
TAU = 0.005
ALPHA = 0.001
MAX_EPISODES = 200
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


class LSDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(LSDQN, self).__init__()
        self.input_layer = nn.Linear(n_observations, 512)
        self.relu1 = nn.ReLU()
        self.hidden_layer = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(512, n_actions)

    def forward(self, x):
        return self.output_layer(self.relu2(self.hidden_layer(self.relu1(self.input_layer(x)))))

    def forward_to_last_hidden(self, x):
        return self.relu2(self.hidden_layer(self.relu1(self.input_layer(x))))

    def chose_action(self, state, action_space_sample, device, epsilon):
        if np.random.rand() < epsilon:
            return torch.tensor([[action_space_sample]], device=device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.forward(state).max(1).indices.view(1, 1)


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = deque([], maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def get_batch(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def add(self, *args):
        self.buffer.append(Experience(*args))


def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def unpack_batch(batch):
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state.cpu(), copy=False)
        states.append(state)
        actions.append(exp.action.cpu())
        rewards.append(exp.reward.cpu())
        dones.append(exp.next_state is None)
        if exp.next_state is None:
            next_states.append(state)       # the result will be masked anyway
        else:
            next_states.append(np.array(exp.next_state.cpu(), copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states, copy=False)


def ls_step(net, tgt_net, batch, gamma, n_srl, lam, device):
    # Calculate FQI matrices
    num_batches = n_srl // 512
    dim = net.hidden_layer.out_features
    num_actions = net.output_layer.out_features

    A = torch.zeros([dim * num_actions, dim * num_actions], dtype=torch.float32).to(device)
    A_bias = torch.zeros([1 * num_actions, 1 * num_actions], dtype=torch.float32).to(device)
    b = torch.zeros([dim * num_actions, 1], dtype=torch.float32).to(device)
    b_bias = torch.zeros([1 * num_actions, 1], dtype=torch.float32).to(device)

    for i in range(num_batches):
        idx = i * 512
        if i == num_batches - 1:
            states, actions, rewards, dones, next_states = unpack_batch(batch[idx:])
        else:
            states, actions, rewards, dones, next_states = unpack_batch(batch[idx: idx + 512])
        states_v = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.BoolTensor(dones).to(device)

        states_features = net.forward_to_last_hidden(states_v)
        # Augmentation
        states_features_aug = torch.zeros([states_features.shape[0], dim * num_actions], dtype=torch.float32).to(device)
        states_features_bias_aug = torch.zeros([states_features.shape[0], 1 * num_actions], dtype=torch.float32).to(
            device)
        for j in range(states_features.shape[0]):
            position = actions_v[j] * dim
            states_features_aug[j, position:position + dim] = states_features.detach()[j, :]
            states_features_bias_aug[j, actions_v[j]] = 1
        states_features_mat = torch.mm(torch.t(states_features_aug), states_features_aug)
        states_features_bias_mat = torch.mm(torch.t(states_features_bias_aug), states_features_bias_aug)
        next_state_values = tgt_net(next_states_v).max(1)[0]

        next_state_values[done_mask] = 0.0

        expected_state_action_values = next_state_values.detach() * gamma + rewards_v  # y_i

        b += torch.mm(torch.t(states_features_aug.detach()), expected_state_action_values.detach().mean(dim=1, keepdim=True))
        b_bias += torch.mm(torch.t(states_features_bias_aug), expected_state_action_values.detach().mean(dim=1, keepdim=True))
        A += states_features_mat.detach()
        A_bias += states_features_bias_mat

    A = (1.0 / n_srl) * A
    A_bias = (1.0 / n_srl) * A_bias
    b = (1.0 / n_srl) * b
    b_bias = (1.0 / n_srl) * b_bias

    original_last_dict = copy.deepcopy(net.output_layer.state_dict())
    last_dict = copy.deepcopy(net.output_layer.state_dict())
    # Calculate retrained weights using FQI closed form solution
    w = last_dict['weight']
    w_b = last_dict['bias']
    num_actions = w.shape[0]
    dim = w.shape[1]
    w = w.view(-1, 1)
    w_b = w_b.view(-1, 1)
    w_srl = torch.mm(torch.inverse(A.detach() + lam * torch.eye(num_actions * dim).to(device)), b.detach() + lam * w.detach())
    w_b_srl = torch.mm(torch.inverse(A_bias.detach() + lam * torch.eye(num_actions * 1).to(device)), b_bias.detach() + lam * w_b.detach())
    w_srl = w_srl.view(num_actions, dim)
    w_b_srl = w_b_srl.squeeze()
    last_dict['weight'] = w_srl.detach()
    last_dict['bias'] = w_b_srl.detach()
    net.output_layer.load_state_dict(last_dict)

    weight_diff = torch.sum((last_dict['weight'] - original_last_dict['weight']) ** 2)
    bias_diff = torch.sum((last_dict['bias'] - original_last_dict['bias']) ** 2)
    total_weight_diff = torch.sqrt(weight_diff + bias_diff)
    print("total weight difference of ls-update: ", total_weight_diff.item())
    print("least-squares step done.")


def decay_epsilon(current_epsilon):
    return max(EPS_END, current_epsilon - EPS_DECAY)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make("Pendulum-v1")
n_actions = env.action_space.n
state, _ = env.reset()
n_observations = len(state)
lsdqn_model = LSDQN(n_observations, n_actions).to(device)
target_model = LSDQN(n_observations, n_actions).to(device)

experience_buffer = ExperienceBuffer(BUFFER_SIZE)
optimizer = optim.Adam(lsdqn_model.parameters(), lr=ALPHA)

while len(experience_buffer) < BUFFER_SIZE - 1:
    state, _ = env.reset()
    done = False
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    while not done:
        action = torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        experience_buffer.add(state, action, reward, next_state)
        state = next_state
        if len(experience_buffer) >= BUFFER_SIZE - 1:
            break

drl_updates = 0
steps_done = 0
episode_durations = []

for episode in range(MAX_EPISODES):
    state, _ = env.reset()
    done = False
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    episode_duration = 0

    while not done:
        action = lsdqn_model.chose_action(state, env.action_space.sample(), device, epsilon)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        experience_buffer.add(state, action, reward, next_state)
        state = next_state

        batch = experience_buffer.get_batch(BATCH_SIZE)
        # Transpose the batch
        batch = Experience(*zip(*batch))
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = lsdqn_model(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_model(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        criterion = nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(lsdqn_model.parameters(), 100)
        optimizer.step()

        # LS-UPDATE STEP
        if drl_updates % n_drl == 0:
            print("performing ls step...")
            batch = experience_buffer.get_batch(n_srl)
            ls_step(lsdqn_model, target_model, batch, GAMMA, len(batch), LAMBDA, device)

        if episode % target_update_freq == 0 and episode != 0:
            target_model.load_state_dict(lsdqn_model.state_dict())

        if done:
            episode_durations.append(episode_duration)
            plot_durations(episode_durations)

        drl_updates += 1
        episode_duration += 1

    epsilon = decay_epsilon(epsilon)

print('Complete')
plot_durations(show_result=True, episode_durations=episode_durations)
plt.ioff()
plt.show()
