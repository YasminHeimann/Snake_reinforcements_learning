from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    An iterable ring-buffer
    """
    def __init__(self, capacity, batch_size=32):
        self.capacity = capacity
        self.memory = []
        self.position = 0

        # for the iterator
        self.batch_size = batch_size
        self.current = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def __iter__(self):
        self.current = 0
        return self

    def __next__(self):
        i_low = self.current * self.batch_size
        self.current += 1
        i_high = self.current * self.batch_size

        if i_high <= len(self.memory):
            return self.memory[i_low:i_high]

        raise StopIteration

    def __len__(self):
        return len(self.memory)

    def flush(self):
        self.memory = []
        self.position = 0
