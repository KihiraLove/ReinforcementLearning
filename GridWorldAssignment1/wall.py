import pkg_resources
import pygame as pg
import config


class Wall(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/wall.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.get_block_dimensions())
        except FileNotFoundError:
            self.image = pg.Surface(config.get_block_dimensions())
            self.image.fill((46, 18, 18))
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE
