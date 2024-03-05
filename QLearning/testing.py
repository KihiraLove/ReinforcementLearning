import time
import numpy as np
from gridworld import GridWorld

world = \
    """
    wwwwwwwwwwwwwwwww
    wa       w     gw
    w      www      w
    wwwww    www  www
    w      www      w
    wwwww    www  www
    w     ww        w
    wwwwwwwwwwwwwwwww
    """

env = GridWorld(world, slip=0.2,
                max_episode_step=1000)  # Beyond max_episode_step interaction, agent get a timelimit error

for i in range(100):  # Number of episodes
    curr_state = env.reset()
    done = False
    while not done:
        env.render()  # [Optional] only if you want to monitor the progress
        action = env.random_action()  # Select by the agent's policy
        next_state, reward, done, info = env.step(action)  # Openai-gym like interface
        print(f"<S,A,R,S'>=<{curr_state},{action},{reward},{next_state}>")
        curr_state = next_state
        time.sleep(0.1)  # Just to see the actions
env.close()  # Must close when rendering is enabled