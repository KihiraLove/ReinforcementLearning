import pygame as pg
import pkg_resources
import config


class Hole(pg.sprite.Sprite):
    def __init__(self, col, row):
        super().__init__()
        fpath = pkg_resources.resource_filename(__name__, 'images/hole.png')
        try:
            self.image = pg.transform.scale(pg.image.load(fpath), config.get_block_dimensions())
        except FileNotFoundError:
            self.image = pg.Surface(config.get_block_dimensions())
            self.image.fill((0, 0, 0))
        self.rect = self.image.get_rect()
        self.pos = pg.Vector2(col, row)
        self.set_pixel_position()
        self.isHole = True

    def set_pixel_position(self):
        self.rect.x = self.pos.x * config.BLOCK_SIZE
        self.rect.y = self.pos.y * config.BLOCK_SIZE

    def draw(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y))

