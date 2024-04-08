from collections import namedtuple, deque

ExperienceFirstLast = namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
END_REWARD = 300


class ExperienceSource:
    def __init__(self, env, agent, gamma):
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.total_rewards = []
        self.total_steps = []
        self.gamma = gamma

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
            obs_len = 1
            states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=1))
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
                    actions[idx] = self.pool[0].action_space.get_batch()
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
                next_state, r, terminated, truncated, _ = env.step(action_n[0])
                is_done = terminated or truncated
                # is_terminated(terminated)
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
                        states[idx] = env.reset()
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