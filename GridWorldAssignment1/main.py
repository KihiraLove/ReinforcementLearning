import time
import config
from agent import Agent
from wall import Wall
from state import State
from goal import Goal
from small_goal import SmallGoal
from hole import Hole
from tree import Tree
import numpy as np
import pygame as pg
from matplotlib import style
from gym.spaces import Box, Discrete
from collections import defaultdict, deque

# w = wall
# a = agent or starting position
# g = goal
# s = small amount of reward
# f = tree
# ' ' = empty tile
# o = hole
start_q_table = None
episode_rewards = []
world = \
    """
    wwwwwwwwwwwwwwwwwwwwwwwww
    wa                      w
    w                       w
    w    fffff              w
    w   ffffff              w
    wffffffffff             w
    w    s   ff             w
    w         f             w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    w                       w
    wff                     w
    wffff                   w
    wfffff                  w
    wffffff                 w
    wfffffff               gw
    wwwwwwwwwwwwwwwwwwwwwwwww
    """

style.use("ggplot")

#########################
# Environment
# My environment is a rewritten version of GridWorld
#########################


class GridWorld:
    def __init__(self, world_string, slip=0.2, log=False):
        self.world = world_string.split('\n    ')[1:-1]
        self.action_map = {0: 'right', 1: 'down', 2: 'left', 3: 'up'}
        self.action_values = [0, 1, 2, 3]
        self.action_size = len(self.action_values)
        self.slip = slip
        self.logging = log
        self._max_epi_step = 1000

        self.columns = len(self.world[0])
        self.rows = len(self.world)
        self.state_color = (50, 100, 10)
        self.render_first = True
        self.policy = {}
        self.episode_step = 0

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
                    self.agent = Agent(col=x, row=y, log=self.logging)
                    self.state_group.add(State(col=x, row=y, color=self.state_color))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.MOVE_COST, 'done': False, 'type': 'norm'}
                    block_count += 1

                elif block_type == 'g':
                    self.goal_group.add(Goal(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': config.GOAL_REWARD, 'done': True, 'type': 'goal'}
                    block_count += 1

                elif block_type == 'f':
                    self.goal_group.add(Tree(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.TREE_COST, 'done': False, 'type': 'norm'}
                    block_count += 1

                elif block_type == 's':
                    self.goal_group.add(SmallGoal(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': config.SMALL_REWARD, 'done': True, 'type': 'goal'}
                    block_count += 1

                elif block_type == 'o':
                    self.state_group.add(Hole(col=x, row=y))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.HOLE_PENALTY, 'done': True, "hole": True, 'type': 'hole'}
                    block_count += 1

                elif block_type == ' ':
                    self.state_group.add(State(col=x, row=y, color=self.state_color))
                    self.state_dict[(x, y)] = {'state': block_count, 'reward': -config.MOVE_COST, 'done': False, 'type': 'norm'}
                    block_count += 1

        self.state_dict = dict(self.state_dict)
        self.state_count = len(self.state_dict)
        # setting action and observation space
        self.action_space = Discrete(self.action_size)
        self.observation_space = Discrete(self.state_count)
        # building environment model
        self.P_sas, self.R_sa = self.build_Model(self.slip)
        self.reset()

    def random_action(self):
        return np.random.choice(self.action_values)

    def format_state(self, response_state):
        return response_state

    def reset(self):
        self.episode_step = 0
        self.agent.re_initialize_agent()
        return self.format_state(
            self.state_dict[(self.agent.initial_position.x, self.agent.initial_position.y)]['state'])

    def get_action_with_probof_slip(self, action):
        individual_slip = self.slip / 3
        prob = [individual_slip for a in self.action_values]
        prob[action] = 1 - self.slip
        act = np.random.choice(self.action_values, p=prob)
        return act

    def step(self, action, testing=False):
        if not testing:
            action = self.get_action_with_probof_slip(action)
        action = self.action_map[action]
        response = self.agent.move(action, self.wall_group, self.state_dict)
        self.episode_step += 1
        if "hole" in response:
            return self.format_state(response['state']), response['reward'], response['done'], {"hole": True}
        elif self.episode_step <= self._max_epi_step:
            return self.format_state(response['state']), response['reward'], response['done'], {}
        else:
            return self.format_state(response['state']), response['reward'], True, {'TimeLimit': True}

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

    def __set_policy(self, policy):
        for i, act in enumerate(policy):
            self.policy[i] = self.action_map[act]
        for s in self.state_group:
            s.change_with_policy(self.state_dict, self.policy)

    def __unset_policy(self):
        self.policy = {}
        for s in self.state_group:
            s.default_state()

    def play_as_human(self, policy=None):
        if policy is not None:
            self.__set_policy(policy)

        pg.init()
        clock = pg.time.Clock()
        done = False
        while not done:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    done = True
                elif event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT:
                        response = self.agent.move('left', self.wall_group, self.state_dict)
                    elif event.key == pg.K_RIGHT:
                        response = self.agent.move('right', self.wall_group, self.state_dict)
                    elif event.key == pg.K_UP:
                        response = self.agent.move('up', self.wall_group, self.state_dict)
                    elif event.key == pg.K_DOWN:
                        response = self.agent.move('down', self.wall_group, self.state_dict)

            self.draw()
            clock.tick(60)
        self.__unset_policy()
        pg.quit()

    def draw(self):
        screen = pg.display.set_mode((self.columns * config.BLOCK_SIZE, self.rows * config.BLOCK_SIZE))
        screen.fill(self.state_color)
        self.wall_group.draw(screen)
        self.state_group.draw(screen)
        self.goal_group.draw(screen)
        self.agent.draw(screen)
        pg.display.update()
        pg.display.flip()

        return screen

    def get_screenshot(self, policy=None):
        if policy is not None:
            self.__set_policy(policy)
        pg.init()
        screen = self.draw()
        image = pg.surfarray.array3d(screen).transpose(1, 0, 2)
        self.__unset_policy()
        pg.quit()
        return image

    def show(self, policy):
        self.play_as_human(policy)

    def build_Model(self, slip):
        P_sas = np.zeros((self.state_count, self.action_size, self.state_count), dtype="float32")
        R_sas = np.zeros((self.state_count, self.action_size, self.state_count), dtype="float32")

        for (col, row), curr_state in self.state_dict.items():
            for act in self.action_values:
                action = self.action_map[act]
                self.agent.set_location(col, row)
                next_state = self.agent.move(action, self.wall_group, self.state_dict)
                P_sas[curr_state["state"], act, next_state["state"]] = 1.0
                R_sas[curr_state["state"], act, next_state["state"]] = next_state["reward"]

        correct = 1 - slip
        ind_slip = slip / 3
        for a in self.action_values:
            other_actions = [oa for oa in self.action_values if oa != a]
            P_sas[:, a, :] = (P_sas[:, a, :] * correct) + (P_sas[:, other_actions, :].sum(axis=1) * ind_slip)

        R_sa = np.multiply(P_sas, R_sas).sum(axis=2)
        return P_sas, R_sa


# Repeat act GridWorld
class RepeatActGridWorld:
    def __init__(self, world_string, slip=0.2, log=False, max_episode_step=1000, repeat_act=4):
        self.repeat_act = repeat_act
        self.state = deque([], maxlen=repeat_act)
        self.env = GridWorld(world_string, slip, log, max_episode_step)
        self.observation_space = Box(low=self.env.observation_space.low.min(),
                                     high=self.env.observation_space.high.max(),
                                     shape=(self.env.observation_space.shape[0] * repeat_act,), dtype='int8')
        self.action_space = self.env.action_space

    def get_state(self):
        return np.array(self.state, dtype='int8').flatten()

    def reset(self):
        s = self.env.reset()
        for i in range(self.repeat_act):
            self.state.append(s)
        return self.get_state()

    def step(self, action, testing=False):
        rewd = 0
        for i in range(self.repeat_act):
            s, r, done, info = self.env.step(action, testing)
            self.state.append(s)
            rewd += r
            if done:
                break
        return self.get_state(), rewd, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


#########################
# Environment ends here
#########################

env = GridWorld(world, slip=0.2)

for i in range(100):  # Number of episodes
    curr_state = env.reset()
    done = False
    while not done:
        env.render()  # [Optional] only if you want to monitor the progress
        action = env.random_action()  # Select by the agent's policy
        next_state, reward, done, info = env.step(action)  # Openai-gym like interface
        print(f"<S,A,R,S'>=<{curr_state},{action},{reward},{next_state}>")
        curr_state = next_state
        time.sleep(0.001)  # Just to see the actions
env.close()  # Must close when rendering is enabled
