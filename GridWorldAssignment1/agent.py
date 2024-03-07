import pygame as pg
import numpy as np
import pkg_resources
from itertools import product
import config


def get_state_symbol(state_type):
    if state_type == 'agent':
        return 0
    elif state_type == 'goal':
        return +2
    elif state_type == 'hole':
        return -2
    elif state_type == 'norm':
        return +1
    elif state_type == 'unknown':
        return -1


class Agent(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/agent.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.get_block_dimensions())
        except FileNotFoundError:
            self.image = pg.Surface(config.get_block_dimensions())
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

    def get_view_state(self, state_dict):
        half_height, half_width = config.get_view_dimensions()
        state_view = np.full((2 * half_height + 1, 2 * half_width + 1),
                             fill_value=get_state_symbol('unknown'), dtype='int8')
        state_view[half_height - 0, half_width - 0] = get_state_symbol("agent")
        for spread in range(1, half_height + 1):
            directional_spread = list(product([-spread, spread], range(-spread, spread + 1))) + list(
                product(range(-(spread - 1), spread), [-spread, spread]))
            # print(dir)
            for modified_x, modified_y in directional_spread:
                try:
                    state_view[half_height + modified_x, half_width + modified_y] = get_state_symbol(
                        state_dict[(self.pos.x + modified_x, self.pos.y + modified_y)]["type"])
                except KeyError:
                    pass
        curr_state_type = get_state_symbol(state_dict[(self.pos.x, self.pos.y)]["type"])
        if np.abs(curr_state_type) == 2:  # terminal states (-2: hole, +2: goal)
            state_view[half_height - 0, half_width - 0] = curr_state_type
        state = np.rot90(np.fliplr(state_view))
        return state

    def re_initialize_agent(self):
        self.pos = pg.Vector2(self.initial_position.x, self.initial_position.y)
        self.set_pixel_position()

    def set_location(self, col, row):
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def draw(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y))
