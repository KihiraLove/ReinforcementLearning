from gridworld import GridWorld
import time
import config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
np.random.seed(42)

world = \
    """
    wwwwwwwwwwwwwwwwwwww
    wa                 w
    w                  w
    w    ooooo         w
    w   oooooo         w
    woooooooooo        w
    w    g   oo        w
    w         o        w
    w                  w
    w                  w
    w                  w
    w                  w
    w                  w
    w                  w
    woo                w
    wooooo             w
    wooooo             w
    wooooooo           w
    woooooooo         gw
    wwwwwwwwwwwwwwwwwwww
    """


alpha = config.LEARNING_RATE
gamma = config.DISCOUNT
epsilon = config.EPSILONS[0]
n = config.N_STEP_PARAMETERS[0]

env = GridWorld(world, slip=0.1, max_episode_step=1000)
episode_rewards = []
q_table = np.zeros((env.state_count, env.action_size))
print("world created")
for episode in range(config.EPISODES):
    state = env.reset()
    render = False
    done = False
    experience_buffer = []
    total_reward = 0

    if episode % config.SHOW_EVERY == 0:
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{config.SHOW_EVERY} ep mean {np.mean(episode_rewards[-config.SHOW_EVERY:])}")
        render = True
    else:
        render = False

    while not done:
        if render:
            env.render()

        if np.random.rand() < epsilon:
            # Exploration
            action = env.random_action()
        else:
            # Exploitation
            action = np.argmax(q_table[state, :])

        next_state, reward, done, info = env.step(action)
        total_reward += reward
        experience = (state, action, reward)
        experience_buffer.append(experience)
        # check is the buffer has enough experience for an N-step update
        if len(experience_buffer) >= n:
            # Retrieve the oldest experience for N-step update
            n_step_experience = experience_buffer[0]
            # Calculate the N-step return
            n_step_return = sum([gamma ** i * experience[2] for i, experience in enumerate(experience_buffer)])
            # Update Q-value for the state-action pair in the N-step experience
            q_table[n_step_experience[0], n_step_experience[1]] += \
                alpha * (n_step_return - q_table[n_step_experience[0], n_step_experience[1]])
            # Remove the oldest experience from buffer
            experience_buffer.pop(0)
        state = next_state
        if render:
            time.sleep(0.001)

    episode_rewards.append(total_reward)
    experience_buffer = []

    if render:
        env.close()

moving_avg = np.convolve(episode_rewards, np.ones((config.SHOW_EVERY,)) / config.SHOW_EVERY, mode="valid")

plt.plot(moving_avg)
plt.ylabel(f"Reward {config.SHOW_EVERY}")
plt.xlabel("Episode number")
plt.savefig("fig.png")
plt.show()
