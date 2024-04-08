import copy
import numpy as np
import torch

epsilon = 0.1
EPS_END = 0.05
EPS_DECAY = 0.05


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


def decay_epsilon(current_epsilon):
    return max(EPS_END, current_epsilon - EPS_DECAY)


def chose_random_action(action_space):
    return np.random.choice(len(action_space))
