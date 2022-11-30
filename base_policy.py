from memory import ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import gym


class BasePolicy:
    # base class for policy implementation
    def __init__(self, buffer_size, gamma, model, actions_space: gym.Space, summery_writer: SummaryWriter, lr):
        self.gamma = gamma
        self.writer = summery_writer   # use this to log your information to tensorboard
        self.model = model
        self.memory = ReplayMemory(capacity=buffer_size)  # example for using this memory - in q_policy.py
        self.action_space = actions_space  # you can sample a random action from here. example in q_policy.py

    def select_action(self, state, epsilon, global_step=None):
        # 'global_step' might be used as time-index for tensorboard recordings.
        raise NotImplementedError()

    def optimize(self, batch_size, global_step=None):
        raise NotImplementedError()

    def record(self, state, action, next_state, reward):
        self.memory.push(state, action, next_state, reward)

    def eval(self):
        self.model = self.model.eval()

    def train(self):
        self.model = self.model.train()

