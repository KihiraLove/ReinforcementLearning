import pygame as pg
import pkg_resources
import config


class State(pg.sprite.Sprite):
    def __init__(self, col, row, color):
        super().__init__()
        self.color = color
        self.image = pg.Surface(config.get_block_dimensions())
        self.image.fill(self.color)
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def default_state(self):
        self.image = pg.Surface(config.get_block_dimensions())
        self.image.fill(self.color)

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE

    def change_with_policy(self, state_dict, policy):  # policy={0:'up',1:'down'} etc
        state = state_dict[(self.pos.x, self.pos.y)]['state']
        optimal_action = policy[state]
        fpath = pkg_resources.resource_filename(__name__, 'images/' + optimal_action + '.png')
        self.image = pg.transform.scale(pg.image.load(fpath), (int(config.BLOCK_SIZE // 2.5),
                                                               int(config.BLOCK_SIZE // 2.5)))
