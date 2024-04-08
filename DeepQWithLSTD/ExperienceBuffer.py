import numpy as np


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity
        self.pos_of_oldest = 0

    def __len__(self):
        return len(self.buffer)

    def get_batch(self, batch_size):
        if len(self.buffer) <= batch_size:
            return self.buffer
        keys = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[key] for key in keys]

    def add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos_of_oldest] = sample
            self.pos_of_oldest = (self.pos_of_oldest + 1) % self.capacity
