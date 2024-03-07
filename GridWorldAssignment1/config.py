from typing import Tuple

GOAL_REWARD: int = 300  # Reward for reaching g state
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
VIEW_SIZE: int = 10  # in pixel, used for rendering


def get_block_dimensions() -> Tuple[int, int]:
    return BLOCK_SIZE, BLOCK_SIZE


def get_view_dimensions() -> Tuple[int, int]:
    return VIEW_SIZE, VIEW_SIZE

