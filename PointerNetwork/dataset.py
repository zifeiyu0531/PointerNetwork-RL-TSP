import numpy as np


class DataGenerator(object):

    # Initialize a DataGenerator
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.max_length = config.max_length

    # Generate random batch for training procedure
    def train_batch(self):
        input_batch = []
        for _ in range(self.batch_size):
            # Generate random TSP instance
            input_ = self.gen_instance()
            # Store batch
            input_batch.append(input_)
        return input_batch

    # Generate random batch for testing procedure
    def test_batch(self):
        # Generate random TSP instance
        input_ = self.gen_instance()
        # Store batch
        input_batch = np.tile(input_, (self.batch_size, 1, 1))
        return input_batch

    # Generate random TSP-TW instance
    def gen_instance(self):
        # Randomly generate (max_length) city
        x = np.random.randint(low=1, high=100, size=(self.max_length, 1))
        y = np.random.randint(low=1, high=100, size=(self.max_length, 1))
        sequence = np.concatenate((x, y), axis=1)

        return sequence
