import numpy as np


class RewardTracker:
    def __init__(self):
        self.total_rewards = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def reward(self, reward, episode, terminated, epsilon):
        self.total_rewards.append(reward)
        mean_reward = np.mean(self.total_rewards[-100:])
        print(f"%d{episode}: done {len(self.total_rewards)} games, mean reward {mean_reward}, eps {epsilon}")
        if terminated:
            print(f"Solved in {episode} episodes!")
            return True
        return False