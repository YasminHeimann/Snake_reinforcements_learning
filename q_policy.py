######################################################
# Yasmin Heimann, hyasmin, 311546915
#
# @description A module that implements Q-Policy class. The class supports select action and
#              optimization based on the Q-Learning algorithm, using a DQN.
#
# @references I used the guidance of the following tutorial:
#              https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
#
######################################################

## IMPORT of packages ##
import torch
import torch.nn.functional as F
import torch.optim as optim
from memory import Transition
from models import SimpleModel, DQN, DQN_C
import random
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym


def create_target_net(network_name):
    """
    Create the correct network for the target net
    :param network_name: the network to create
    :return: the target model
    """
    if network_name == 'simple':
        return SimpleModel()
    elif network_name == 'dqn_linear':
        return DQN()
    elif network_name == 'dqn_conv':
        return DQN_C()


class QPolicy(BasePolicy):
    # Q Policy class that implements the q-learning algorithm
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter,
                 lr, network_name):
        super(QPolicy, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.target_net = create_target_net(network_name)
        self.target_net.load_state_dict(self.model.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.cur_loss = 0

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        random_number = random.random()
        # choose exploration vs. exploitation
        if random_number > epsilon:  # exploitation
            with torch.no_grad():
                return self.model(state).argmax().item()
        else:  # exploration
            return self.action_space.sample()  # return action randomly

    def optimize(self, batch_size, global_step=None):
        """
        Optimizes the current Q-policy by DQN optimization, going over the past predictions, and comparing them
        to approximated optimal Q-policy, which is the target network.
        """
        loss_avg, total = 0, 0
        if len(self.memory) < batch_size:
            return None

        self.memory.batch_size = batch_size
        for transitions_batch in self.memory:
            # transform list of tuples into a tuple of lists.
            # explanation here: https://stackoverflow.com/a/19343/3343043
            batch = Transition(*zip(*transitions_batch))

            state_batch = torch.cat(batch.state)
            next_state_batch = torch.cat(batch.next_state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # get the predicted q value from the model on the given state
            state_action_q_value = self.model(state_batch).gather(dim=1, index=action_batch.unsqueeze(1))
            next_state_value = self.target_net(next_state_batch).max(1)[0].detach()

            # Compute the expected Q values (Bellman's)
            expected_state_action_value = (next_state_value * self.gamma) + reward_batch

            # Compute Huber loss
            loss = F.smooth_l1_loss(state_action_q_value, expected_state_action_value.unsqueeze(1))
            loss_avg += loss.item()
            total += 1

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
        # update the gradients of the target net
        self.target_net.load_state_dict(self.model.state_dict())
        self.cur_loss = loss_avg / total

    def get_current_loss(self):
        """
        :return: the average loss that was calculated in the last optimization step
        """
        return self.cur_loss
