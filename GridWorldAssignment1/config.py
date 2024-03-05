from typing import Tuple

GOAL_REWARD: int = 100
SMALL_REWARD: int = 3
TREE_COST: int = 10
MOVE_COST: int = 1
HOLE_PENALTY: int = 100
SIZE: int = 25
EPISODES: int = 25000
epsilon: float = 0.9
EPSILON_DECAY: float = 0.9998
SHOW_EVERY: int = 2000
LEARNING_RATE: float = 0.1
DISCOUNT: float = 0.95
BLOCK_SIZE: int = 25  # in pixel
VIEW_SIZE: int = 10  # in pixel


def get_block_dimensions() -> Tuple[int, int]:
    return BLOCK_SIZE, BLOCK_SIZE


def get_view_dimensions() -> Tuple[int, int]:
    return VIEW_SIZE, VIEW_SIZE

