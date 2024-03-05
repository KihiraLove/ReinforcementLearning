import numpy as np

class NStepQLearningAgent:
    def __init__(self, num_states, num_actions, alpha=0.1, gamma=0.9, epsilon=0.1, n=1):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Epsilon for epsilon-greedy policy
        self.n = n  # N-step parameter

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))
        # Buffer to store experiences for N-step updates
        self.experience_buffer = []

    def choose_action(self, state):
        # Epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            return np.random.choice(self.num_actions)
        else:
            # Exploitation: choose the action with the highest Q-value
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        # Store the current experience in the buffer
        self.experience_buffer.append((state, action, reward))

        # Check if the buffer has enough experiences for an N-step update
        if len(self.experience_buffer) >= self.n:
            # Retrieve the oldest experience for N-step update
            n_step_experience = self.experience_buffer[0]

            # Calculate the N-step return
            n_step_return = sum([self.gamma ** i * exp[2] for i, exp in enumerate(self.experience_buffer)])

            # Update Q-value for the state-action pair in the N-step experience
            self.q_table[n_step_experience[0], n_step_experience[1]] += \
                self.alpha * (n_step_return - self.q_table[n_step_experience[0], n_step_experience[1]])

            # Remove the oldest experience from the buffer
            self.experience_buffer.pop(0)

    def train(self, num_episodes, env):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            while True:
                action = self.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # Update Q-table using N-step Q-learning
                self.update_q_table(state, action, reward, next_state)

                state = next_state

                if done:
                    # Clear the experience buffer at the end of the episode
                    self.experience_buffer = []
                    break

            print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

# Example usage:
# Assume you have an environment 'env' with num_states and num_actions
# env = YourEnvironment()
# num_states = env.num_states
# num_actions = env.num_actions

# Create N-step Q-learning agents for different values of epsilon
# n_values = [1, 2, 3]
# epsilon_values = [0.1, 0.2, 0.3]

# for n in n_values:
#     for epsilon in epsilon_values:
#         q_agent = NStepQLearningAgent(num_states, num_actions, epsilon=epsilon, n=n)
#         num_episodes = 1000
#         q_agent.train(num_episodes, env)
