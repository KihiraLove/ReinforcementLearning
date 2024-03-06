from typing import Tuple

GOAL_REWARD: int = 100
SMALL_REWARD: int = 5
TREE_COST: int = 50
MOVE_COST: int = 1
HOLE_PENALTY: int = 1000

TIMEOUT = 1000
TIMEOUT_PENALTY = 200

EPISODES: int = 1000
SHOW_EVERY: int = 100

EPSILON_DECAY: float = 0.9998
LEARNING_RATE: float = 0.1
DISCOUNT: float = 0.95
EPSILONS: Tuple[float, float, float] = (0.1, 0.2, 0.3)
N_STEP_PARAMETERS: Tuple[int, int, int] = (1, 2, 3)

BLOCK_SIZE: int = 25  # in pixel
VIEW_SIZE: int = 10  # in pixel


def get_block_dimensions() -> Tuple[int, int]:
    return BLOCK_SIZE, BLOCK_SIZE


def get_view_dimensions() -> Tuple[int, int]:
    return VIEW_SIZE, VIEW_SIZE

