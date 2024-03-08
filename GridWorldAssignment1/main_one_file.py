import os
import pkg_resources
import numpy as np
import pygame as pg
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import style
from gym.spaces import Discrete
from collections import defaultdict

style.use("seaborn-v0_8-whitegrid")
np.random.seed(42)

#########################################################
# Author: Domonkos Kertész (doker24)
# Date: 07.03.2024.
#
# This project can be found @ https://github.com/KihiraLove/ReinforcementLearning/tree/main/GridWorldAssignment1
#
#                   Configuration: lines 103 - 128
#                     Environment: lines 134 - 423
#         Functions for algorithm: lines 430 - 128
#         Q-Learning specifically: lines 440 - 458
#                          Worlds: lines 526 - 581
#                        Training: lines 584 - 610
# Calculating values and plotting: lines 612 - 679
#
# The environment can load and render with or without the directory of images
# Running this file will result in training 9 configurations for my Q-learning algorithm
# Two png files will be created for all configurations in the working directory (1 for learning curve over 10 iterations, 1 for showing why episodes ended)
#
# !Running this file will result in generating the following file structure if config.LOGGING = True!
# root
#   └── world1
#          └── 1_step
#               └── 0.1
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.2
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.3
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#          └── 2_step
#               └── 0.1
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.2
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.3
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#          └── 3_step
#               └── 0.1
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.2
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.3
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#   └── world2
#          └── 1_step
#               └── 0.1
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.2
# #                   └── [ ending_causes1.png, ... , ending_causes10.png ]
# #                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.3
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#          └── 2_step
#               └── 0.1
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.2
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.3
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#          └── 3_step
#               └── 0.1
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.2
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#               └── 0.3
#                   └── [ ending_causes1.png, ... , ending_causes10.png ]
#                   └── [ learning_curve1.png, ... , learning_curve10.png ]
#########################################################


#########################################################
# Config
# This class holds the input parameters for the algorithm and values for rendering
#########################################################
class config:
    LOGGING: bool = False  # If this is true the script will create and save about 400 plots

    GOAL_REWARD: int = 1000  # Reward for reaching g state
    SMALL_REWARD: int = 5  # Reward for reaching s state
    TREE_COST: int = 10  # Cost to cross f state
    MOVE_COST: int = 1  # Cost to cross ' ' state
    HOLE_PENALTY: int = 1000  # Penalty for reaching o state

    # At TIMEOUT = 1000 steps the agent is greatly disinclined from exploring, and chose to end episodes quickly
    # with world2 the agent prioritizes ending the episode quickly with a hole rather than exploring
    TIMEOUT = 1500  # Number of steps in an episode before timeout penalty
    TIMEOUT_PENALTY = 100  # Penalty for timing out

    ITERATIONS: int = 10  # NUmber of training iterations
    EPISODES: int = 500  # Number of episodes
    EPISODES_IN_GROUP: int = 5  # Number of episodes in a group

    LEARNING_RATE: float = 0.1
    DISCOUNT: float = 0.95
    EPSILONS: Tuple[float, float, float] = (0.1, 0.2, 0.3)
    N_STEP_PARAMETERS: Tuple[int, int, int] = (1, 2, 3)

    BLOCK_SIZE: int = 25  # in pixel, used for rendering

    BLOCK_DIMENSIONS: Tuple[int, int] = (BLOCK_SIZE, BLOCK_SIZE)


#########################################################
# Environment built with pygame
# My environment is a rewritten version of GridWorld from https://github.com/prasenjit52282/GridWorld
#########################################################
class Agent(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/agent.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.BLOCK_DIMENSIONS)
        except FileNotFoundError:
            self.image = pg.Surface(config.BLOCK_DIMENSIONS)
            self.image.fill((0, 51, 255))
        self.rect = self.image.get_rect()
        self.initial_position = pg.Vector2(col, row)

        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE

    def move(self, direction, walls, state_dict):
        previous_position = pg.Vector2(self.pos.x, self.pos.y)
        if hasattr(state_dict[(previous_position.x, previous_position.y)], "isHole"):
            self.pos = pg.Vector2(previous_position.x, previous_position.y)
        elif direction == 'down':
            self.pos += pg.Vector2(0, 1)
        elif direction == 'up':
            self.pos += pg.Vector2(0, -1)
        elif direction == 'right':
            self.pos += pg.Vector2(1, 0)
        elif direction == 'left':
            self.pos += pg.Vector2(-1, 0)
        for wall in walls:
            if self.pos == wall.pos:
                self.pos = pg.Vector2(previous_position.x, previous_position.y)
                break
        self.set_pixel_position()
        next_state = state_dict[(self.pos.x, self.pos.y)]
        return next_state

    def re_initialize_agent(self):
        self.pos = pg.Vector2(self.initial_position.x, self.initial_position.y)
        self.set_pixel_position()

    def set_location(self, col, row):
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def draw(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y))


class Goal(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/goal.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.BLOCK_DIMENSIONS)
        except FileNotFoundError:
            self.image = pg.Surface(config.BLOCK_DIMENSIONS)
            self.image.fill((255, 162, 0))
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE


class Hole(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/hole.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.BLOCK_DIMENSIONS)
        except FileNotFoundError:
            self.image = pg.Surface(config.BLOCK_DIMENSIONS)
            self.image.fill((0, 0, 0))
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()
        self.isHole = True

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE


class SmallGoal(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/small.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.BLOCK_DIMENSIONS)
        except FileNotFoundError:
            self.image = pg.Surface(config.BLOCK_DIMENSIONS)
            self.image.fill((255, 204, 0))
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE


class Tree(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/tree.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.BLOCK_DIMENSIONS)
        except FileNotFoundError:
            self.image = pg.Surface(config.BLOCK_DIMENSIONS)
            self.image.fill((17, 26, 12))
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE


class Wall(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/wall.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.BLOCK_DIMENSIONS)
        except FileNotFoundError:
            self.image = pg.Surface(config.BLOCK_DIMENSIONS)
            self.image.fill((46, 18, 18))
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE


class State(pg.sprite.Sprite):
    def __init__(self, col, row, color):
        super().__init__()
        self.color = color
        self.image = pg.Surface(config.BLOCK_DIMENSIONS)
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def default_state(self):
        self.image = pg.Surface(config.BLOCK_DIMENSIONS)
        self.image.fill(self.color)

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE


class GridWorld:
    def __init__(self, world_string, slip):
        self.world = world_string.split('\n    ')[1:-1]
        self.action_map = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
        self.action_values = [0, 1, 2, 3]
        self.action_size = len(self.action_values)
        self.slip = slip

        self.screen = None

        self.columns = len(self.world[0])
        self.rows = len(self.world)
        self.state_color = (50, 100, 10)
        self.render_first = True
        self.policy = {}

        self.wall_group = pg.sprite.Group()
        self.state_group = pg.sprite.Group()
        self.state_dict = defaultdict(lambda: 0)
        self.goal_group = pg.sprite.Group()

        block_count = 0
        for y, et_row in enumerate(self.world):
            for x, block_type in enumerate(et_row):

                if block_type == 'w':
                    self.wall_group.add(Wall(col=x, row=y))

                elif block_type == 'a':
                    self.agent = Agent(col=x, row=y)
                    self.state_group.add(State(col=x, row=y, color=self.state_color))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.MOVE_COST, 'done': False,
                                               'type': 'norm'}
                    block_count += 1

                elif block_type == 'g':
                    self.goal_group.add(Goal(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': config.GOAL_REWARD, 'done': True,
                                               'type': 'goal'}
                    block_count += 1

                elif block_type == 'f':
                    self.state_group.add(Tree(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.TREE_COST, 'done': False,
                                               'type': 'norm'}
                    block_count += 1

                elif block_type == 's':
                    self.goal_group.add(SmallGoal(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': config.SMALL_REWARD, 'done': True,
                                               'type': 'goal'}
                    block_count += 1

                elif block_type == 'o':
                    self.state_group.add(Hole(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.HOLE_PENALTY, 'done': True,
                                               "hole": True, 'type': 'hole'}
                    block_count += 1

                elif block_type == ' ':
                    self.state_group.add(State(col=x, row=y, color=self.state_color))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.MOVE_COST, 'done': False,
                                               'type': 'norm'}
                    block_count += 1

        self.state_dict = dict(self.state_dict)
        self.state_count = len(self.state_dict)
        # setting action and observation space
        self.action_space = Discrete(self.action_size)
        self.observation_space = Discrete(self.state_count)
        # building environment model
        self.P_sas, self.R_sa = self.build_model(self.slip)
        self.reset()

    def reset(self):
        self.agent.re_initialize_agent()
        return self.state_dict[(self.agent.initial_position.x, self.agent.initial_position.y)]['state']

    def step(self, action):
        action = self.action_map[action]
        response = self.agent.move(action, self.wall_group, self.state_dict)
        if "hole" in response:
            return response['state'], response['reward'], response['done'], {"hole": True}
        else:
            return response['state'], response['reward'], response['done'], {}

    def render(self):
        if self.render_first:
            pg.init()
            self.screen = pg.display.set_mode((self.columns * config.BLOCK_SIZE, self.rows * config.BLOCK_SIZE))
            self.render_first = False
        self.screen.fill(self.state_color)
        self.wall_group.draw(self.screen)
        self.state_group.draw(self.screen)
        self.goal_group.draw(self.screen)
        self.agent.draw(self.screen)
        pg.display.update()
        pg.display.flip()

    def close(self):
        self.render_first = True
        pg.quit()

    def build_model(self, slip):
        state_action_state_probability_array = np.zeros((self.state_count, self.action_size, self.state_count),
                                                        dtype="float32")
        state_action_state_reward_array = np.zeros((self.state_count, self.action_size, self.state_count),
                                                   dtype="float32")

        for (column, row), current_state in self.state_dict.items():
            for act in self.action_values:
                action = self.action_map[act]
                self.agent.set_location(column, row)
                next_state = self.agent.move(action, self.wall_group, self.state_dict)
                state_action_state_probability_array[current_state["state"], act, next_state["state"]] = 1.0
                state_action_state_reward_array[current_state["state"], act, next_state["state"]] = next_state["reward"]

        correct = 1 - slip
        ind_slip = slip / 3
        for a in self.action_values:
            other_actions = [oa for oa in self.action_values if oa != a]
            state_action_state_probability_array[:, a, :] = (state_action_state_probability_array[:, a,
                                                             :] * correct) + (state_action_state_probability_array[:,
                                                                              other_actions, :].sum(axis=1) * ind_slip)

        state_action_state_reward_array = np.multiply(state_action_state_probability_array,
                                                      state_action_state_reward_array).sum(axis=2)
        return state_action_state_probability_array, state_action_state_reward_array

#########################################################
# Environment ends here
#########################################################


def chose_action(s, q_table, eps, env):
    # Epsilon greedy policy
    if np.random.rand() < eps:
        # Exploration
        return np.random.choice(env.action_size)
    else:
        # Exploitation
        return np.argmax(q_table[s, :])


def update_q_table(exp_buffer, gamma, alpha, q_table, next_state):
    # Calculate N-step count from length of experience buffer
    n = len(exp_buffer)
    # Retrieve the oldest experience for N-step update
    n_step_experience = exp_buffer[0]
    # Calculate the N-step return
    n_step_return = sum([gamma ** i * exp[2] for i, exp in enumerate(exp_buffer)])
    # Update Q-value for the state-action pair in the oldest experience
    if next_state is not None:
        q_table[n_step_experience[0], n_step_experience[1]] += \
            alpha * (n_step_return + gamma ** n * np.max(q_table[next_state, :])
                     - q_table[n_step_experience[0], n_step_experience[1]])
    else:
        # Update Q-values for the last state-action pair/pairs if the agent reaches a goal state
        q_table[n_step_experience[0], n_step_experience[1]] += alpha * (
                n_step_return - q_table[n_step_experience[0], n_step_experience[1]])
    # Remove the oldest experience from buffer
    exp_buffer.pop(0)
    return q_table, exp_buffer


def train(world, n, gamma, alpha, epsilon, iteration):
    env = GridWorld(world[1], epsilon)
    q_table = np.zeros((env.state_count, env.action_size))
    episode_rewards = []
    episode_ending_cause = [0, 0, 0]
    steps = []
    for episode in range(config.EPISODES):
        state = env.reset()
        done = False
        experience_buffer = []
        total_reward = 0

        if episode % config.EPISODES_IN_GROUP == 0 and episode != 0 and config.LOGGING:
            print(f"on # {episode}, epsilon: {epsilon}")
            print(f"{config.EPISODES_IN_GROUP} ep mean {np.mean(episode_rewards[-config.EPISODES_IN_GROUP:])}")

        step = 0
        while not done:
            action = chose_action(state, q_table, epsilon, env)
            next_state, reward, done, _ = env.step(action)
            # Log the reason for ending the episode
            if done:
                steps.append(step)
                if reward == config.SMALL_REWARD:
                    episode_ending_cause[0] += 1
                elif reward == config.GOAL_REWARD:
                    episode_ending_cause[1] += 1
                elif reward == -config.HOLE_PENALTY:
                    episode_ending_cause[2] += 1
            # Apply timeout penalty for each step after timeout
            if step > config.TIMEOUT and False:
                reward -= config.TIMEOUT_PENALTY

            total_reward += reward
            experience = (state, action, reward)
            experience_buffer.append(experience)
            # Check is the buffer has enough experience for an N-step update
            if len(experience_buffer) >= n:
                q_table, experience_buffer = update_q_table(experience_buffer, gamma, alpha, q_table, next_state)
                if done:
                    # Update the remaining states from the buffer after reaching a goal state
                    while len(experience_buffer) > 0:
                        q_table, experience_buffer = update_q_table(experience_buffer, gamma, alpha, q_table, None)

            state = next_state
            step += 1

        episode_rewards.append(total_reward)
        experience_buffer.clear()

    print("Training configuration:", world[0], "N-step:", n, "epsilon:", epsilon, "iteration:", iteration+1)
    print("Number of episodes ended by reaching the Goal state:", episode_ending_cause[1])
    print("Number of episodes ended by reaching the Small Goal state:", episode_ending_cause[0])
    print("Number of episodes ended by reaching the Hole state:", episode_ending_cause[2])
    print("############################################")
    return episode_rewards, episode_ending_cause, steps


# w = wall
# a = agent or starting position
# g = goal
# s = small amount of reward
# f = tree/forest (High movement cost)
# ' ' = empty tile (Normal movement cost)
# o = hole
world_1 = \
    """
    wwwwwwwwwwwwwwwwwwwwwwwww
    w                 ffffffw
    w   a              fffffw
    w                   ffffw
    wffffff              fffw
    wfffffffff            ffw
    wfffffffffff            w
    wffffffffffff           w
    wffffffffffff           w
    w        ffff           w
    w    s    fff           w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    wff                  gggw
    wffff               ggggw
    wfffff             gggggw
    wffffff           ggggggw
    wfffffff         gggggggw
    wwwwwwwwwwwwwwwwwwwwwwwww
    """

world_2 = \
    """
    wwwwwwwwwwwwwwwwwwwwwwwww
    wa               ooooooow
    w                 oooooow
    wooooo              oooow
    woooooo               oow
    woooooo                 w
    w oooo                  w
    w                       w
    w                       w
    w                       w
    w                 oooooow
    w                 oo    w
    w                       w
    woooooo                 w
    w                       w
    w       ooooo         oow
    w                       w
    w       oooooo          w
    w         ooooo         w
    wff       ooooo         w
    wffff      ooo          w
    wfffff                  w
    wffffff                 w
    wfffffff               gw
    wwwwwwwwwwwwwwwwwwwwwwwww
    """

worlds = [("world1", world_1), ("world2", world_2)]
for world in worlds:
    for n in config.N_STEP_PARAMETERS:
        for epsilon in config.EPSILONS:
            all_episode_rewards = []
            all_ending_causes = []
            all_step_count = []

            #  Check for directory tree, create directories in current working directory
            if config.LOGGING:
                current_directory = os.getcwd()
                world_dir = os.path.join(current_directory, "world" + str(worlds.index(world) + 1))
                step_dir = os.path.join(world_dir, str(n) + "_step")
                epsilon_dir = os.path.join(step_dir, str(epsilon))
                if not os.path.exists(world_dir):
                    os.makedirs(world_dir)
                if not os.path.exists(step_dir):
                    os.makedirs(step_dir)
                if not os.path.exists(epsilon_dir):
                    os.makedirs(epsilon_dir)

            for it in range(config.ITERATIONS):
                #  Train algorithm and collect data for current iteration
                ep_rew, end_cause, st = train(world, n, config.DISCOUNT, config.LEARNING_RATE, epsilon, it)
                all_episode_rewards.append(ep_rew)
                all_ending_causes.append(end_cause)
                all_step_count.append(st)

                if config.LOGGING:
                    #  Save the learning curve of a given iteration to the respective folder
                    moving_avg = np.convolve(ep_rew, np.ones((config.EPISODES_IN_GROUP,)) / config.EPISODES_IN_GROUP, mode="valid")
                    plt.plot(moving_avg, color='red', label="Learning curve")
                    plt.title(f"Learning curve of world{worlds.index(world) + 1} {n}-step epsilon: {epsilon} iteration: {it+1}")
                    plt.ylabel(f"Reward {config.EPISODES_IN_GROUP}")
                    plt.xlabel("Episode number")
                    plt.savefig(epsilon_dir + f"/learning_curve_{it+1}.png")
                    plt.close()

                    #  Save the causes of the episodes ending of a given iteration to the respective folder
                    ending_cause_names = ["goal", "small goal", "hole"]
                    ending_causes = [end_cause[1], end_cause[0], end_cause[2]]
                    plt.bar(ending_cause_names, ending_causes, width=1, edgecolor="white", linewidth=0.7)
                    plt.title(f"Episode end causes of world{worlds.index(world) + 1} {n}-step epsilon: {epsilon} iteration: {it+1}")
                    plt.ylabel("Number of causes")
                    plt.xlabel("Type of cause")
                    plt.savefig(epsilon_dir + f"/ending_causes_{it+1}.png")
                    plt.close()

            mean_rewards = []
            for iteration_rewards in all_episode_rewards:
                groups = []
                # Split the episode rewards into groups according to the configuration, get the mean of each group
                for i in range(0, len(iteration_rewards), config.EPISODES_IN_GROUP):
                    groups.append(iteration_rewards[i:i + config.EPISODES_IN_GROUP])
                mean_group_rewards = [np.mean(group) for group in groups]
                mean_rewards.append(mean_group_rewards)

            #  Stack the means and calculate standard error
            stacked_means = np.vstack(mean_rewards)
            standard_errors = np.std(stacked_means, axis=0) / np.sqrt(stacked_means.shape[0])
            #  Plot the learning curve over 10 iterations, rewards grouped, and standard error
            plt.plot(np.arange(1, len(mean_rewards[0]) + 1) * config.EPISODES_IN_GROUP, np.mean(mean_rewards, axis=0), label='Mean', color='red')
            plt.fill_between(np.arange(1, len(mean_rewards[0]) + 1) * config.EPISODES_IN_GROUP,
                             np.mean(mean_rewards, axis=0) - standard_errors,
                             np.mean(mean_rewards, axis=0) + standard_errors,
                             color='black', alpha=0.2, label='Standard Error')
            plt.xlabel("Episodes")
            plt.ylabel("Total Reward")
            plt.title(f"Mean Reward with Standard Error of\nworld{worlds.index(world) + 1} {n}-step epsilon: {epsilon}")
            plt.legend()
            plt.savefig(f"Learning_curve_world{worlds.index(world) + 1}_{n}-step_epsilon_{epsilon}.png")
            plt.close()

            if config.LOGGING:
                #  Calculate of the means of why episodes ended throughout the 10 iterations
                mean_of_goals = 0
                mean_of_small_goals = 0
                mean_of_holes = 0
                for i in range(config.ITERATIONS):
                    mean_of_goals += all_ending_causes[i][1]
                    mean_of_small_goals += all_ending_causes[i][0]
                    mean_of_holes += all_ending_causes[i][2]

                mean_of_goals /= config.ITERATIONS
                mean_of_small_goals /= config.ITERATIONS
                mean_of_holes /= config.ITERATIONS
                ending_cause_names = ["goal", "small goal", "hole"]
                ending_causes = [mean_of_goals, mean_of_small_goals, mean_of_holes]
                # Plot the causes of episode ends on a bar plot
                plt.bar(ending_cause_names, ending_causes, width=1, edgecolor="white", linewidth=0.7)
                plt.title(
                    f"Mean of episode end causes of world{worlds.index(world) + 1} {n}-step epsilon: {epsilon}")
                plt.ylabel("Number of causes")
                plt.xlabel("Type of cause")
                plt.savefig(f"Ending_cause_world{worlds.index(world) + 1}_{n}-step_epsilon_{epsilon}.png")
                plt.close()
