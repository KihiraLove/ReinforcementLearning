import gymnasium as gym
import torch
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple, deque
from tensorboardX import SummaryWriter
import time
import os
import copy
import numpy as np
import sys

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 0.05
DECAY_EVERY = 1000
TAU = 0.005
ALPHA = 0.001
BUFFER_SIZE = 5000
n_drl = 5000
LAMBDA = 1
END_REWARD = 300
epsilon = 0.1

ExperienceFirstLast = namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])


class LSDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(LSDQN, self).__init__()

        print(input_shape)
        print(input_shape[0])
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


class TargetModel:
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())


class DQNAgent:
    def __init__(self, dqn_model, device):
        self.dqn_model = dqn_model
        self.device = device

    def __call__(self, states, agent_states=None):
        if agent_states is None:
            agent_states = [None] * len(states)

        if len(states) == 1:
            np_states = states[0][0]
        else:
            np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
        states = torch.tensor(np_states)
        if torch.is_tensor(states):
            states = states.to(self.device)
        q_v = self.dqn_model(states)
        q = q_v.data.cpu().numpy()
        actions = chose_action(q)
        return actions, agent_states


class ExperienceSource:
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in self._iterate_experience():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)

    def _iterate_experience(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(None)

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                for idx in range(states_actions):
                    action = states_actions[idx]
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]

            grouped_actions = []
            cur_ofs = 0
            for g_len in env_lens:
                grouped_actions.append(actions[cur_ofs:cur_ofs + g_len])
                cur_ofs += g_len

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, terminated, truncated, _ = env.step(action_n)
                    is_done_n = terminated[0] or truncated[0]
                    is_done_n = [is_done_n]
                    if terminated:
                        r_n = [END_REWARD]
                else:
                    next_state, r, terminated, truncated, _ = env.step(action_n[0])
                    is_done = terminated or truncated
                    if terminated:
                        r = END_REWARD
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done:
                        # generate tail of history
                        while len(history) >= 1:
                            yield tuple(history)
                            history.popleft()
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)


class RewardTracker:
    def __init__(self, stop_reward):
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        pass

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (
            frame, len(self.total_rewards), mean_reward, speed, epsilon_str
        ))
        sys.stdout.flush()
        if mean_reward > self.stop_reward:
            print("Solved in %d frames!" % frame)
            return True
        return False


def calc_loss_dqn(batch, net, tgt_net, gamma, device):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)       # the result will be masked anyway
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), \
           np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


def ls_step(net, tgt_net, batch, gamma, n_srl, lam, m_batch_size, device):
    # Calculate FQI matrices
    num_batches = n_srl // m_batch_size
    dim = net.fc1.out_features
    num_actions = net.fc2.out_features

    A = torch.zeros([dim * num_actions, dim * num_actions], dtype=torch.float32).to(device)
    A_bias = torch.zeros([1 * num_actions, 1 * num_actions], dtype=torch.float32).to(device)
    b = torch.zeros([dim * num_actions, 1], dtype=torch.float32).to(device)
    b_bias = torch.zeros([1 * num_actions, 1], dtype=torch.float32).to(device)

    for i in range(num_batches):
        idx = i * m_batch_size
        if i == num_batches - 1:
            states, actions, rewards, dones, next_states = unpack_batch(batch[idx:])
        else:
            states, actions, rewards, dones, next_states = unpack_batch(batch[idx: idx + m_batch_size])
        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

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

        b += torch.mm(torch.t(states_features_aug.detach()), expected_state_action_values.detach().view(-1, 1))
        b_bias += torch.mm(torch.t(states_features_bias_aug), expected_state_action_values.detach().view(-1, 1))
        A += states_features_mat.detach()
        A_bias += states_features_bias_mat

    A = (1.0 / n_srl) * A
    A_bias = (1.0 / n_srl) * A_bias
    b = (1.0 / n_srl) * b
    b_bias = (1.0 / n_srl) * b_bias

    w_last_before = copy.deepcopy(net.fc2.state_dict())
    w_last_dict = copy.deepcopy(net.fc2.state_dict())
    # Calculate retrained weights using FQI closed form solution
    w = w_last_dict['weight']
    w_b = w_last_dict['bias']
    num_actions = w.shape[0]
    dim = w.shape[1]
    w = w.view(-1, 1)
    w_b = w_b.view(-1, 1)
    w_srl = torch.mm(torch.inverse(A.detach() + lam * torch.eye(num_actions * dim).to(device)),
                     b.detach() + lam * w.detach())
    w_b_srl = torch.mm(torch.inverse(A_bias.detach() + lam * torch.eye(num_actions * 1).to(device)),
                       b_bias.detach() + lam * w_b.detach())
    w_srl = w_srl.view(num_actions, dim)
    w_b_srl = w_b_srl.squeeze()
    w_last_dict['weight'] = w_srl.detach()
    w_last_dict['bias'] = w_b_srl.detach()
    net.fc2.load_state_dict(w_last_dict)

    weight_diff = torch.sum((w_last_dict['weight'] - w_last_before['weight']) ** 2)
    bias_diff = torch.sum((w_last_dict['bias'] - w_last_before['bias']) ** 2)
    total_weight_diff = torch.sqrt(weight_diff + bias_diff)
    print("total weight difference of ls-update: ", total_weight_diff.item())
    print("least-squares step done.")


def chose_action(actions):
    if np.random.rand() < epsilon:
        return np.random.choice(len(actions))
    else:
        return np.argmax(actions)


def decay_epsilon(epsilon):
    return max(EPS_END, epsilon - EPS_DECAY)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
env = gym.make("MountainCar-v0")

target_update_freq = 1000

save_freq = 50000
n_srl = BATCH_SIZE  # size of batch in SRL step

print("using ls-dqn with lambda:", str(LAMBDA))

print(env.observation_space.shape)
print(env.action_space.n)
lsdqn_model = LSDQN(env.observation_space.shape, env.action_space.n).to(device)
target_model = TargetModel(lsdqn_model)

agent = DQNAgent(lsdqn_model, device)

exp_source = ExperienceSource(env, agent, GAMMA, steps_count=1)
buffer = ExperienceReplayBuffer(exp_source, BUFFER_SIZE)
optimizer = optim.Adam(lsdqn_model.parameters(), lr=ALPHA)

frame_idx = 0
drl_updates = 0

with RewardTracker(END_REWARD) as reward_tracker:
    while True:
        frame_idx += 1
        buffer.populate(1)
        epsilon = decay_epsilon(epsilon)

        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            if reward_tracker.reward(new_rewards[0], frame_idx, epsilon):
                break

        if len(buffer) < BUFFER_SIZE:
            continue

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_v = calc_loss_dqn(batch, lsdqn_model, target_model.target_model, GAMMA, device)
        loss_v.backward()
        optimizer.step()
        drl_updates += 1

        # LS-UPDATE STEP
        if (drl_updates % n_drl == 0) and (len(buffer) >= n_srl):
            print("performing ls step...")
            batch = buffer.sample(n_srl)
            ls_step(lsdqn_model, target_model.target_model, batch, GAMMA, len(batch), LAMBDA, BATCH_SIZE, device)

        if frame_idx % target_update_freq == 0:
            target_model.sync()
