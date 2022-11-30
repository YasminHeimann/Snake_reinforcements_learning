######################################################
# Yasmin Heimann, hyasmin, 311546915
#
# @description A module that implements Policy Gradient (MC) class. The class supports select action and
#              optimization based on the Monte Carlo Policy Gradient algorithm, using a NN to predict Q.
#
######################################################

## IMPORT of packages ##
import torch
import torch.optim as optim
from memory import Transition
from torch.utils.tensorboard import SummaryWriter
from base_policy import BasePolicy
import gym
from torch.distributions import Categorical


class PolicyGradient(BasePolicy):
    # Policy Gradient class that implements the REINFORCE algorithm
    def __init__(self, buffer_size, gamma, model, action_space: gym.Space, summery_writer: SummaryWriter,
                 lr, network_name):
        super(PolicyGradient, self).__init__(buffer_size, gamma, model, action_space, summery_writer, lr)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr)
        self.alpha = 1
        self.cur_loss = 0

    def select_action(self, state, epsilon, global_step=None):
        """
        select an action to play
        :param state: 1x9x9x10 (nhwc), n=1, h=w=9, c=10 (types of thing on the board, one hot encoding)
        :param epsilon: epsilon...
        :param global_step: used for tensorboard logging
        :return: return single action as integer (0, 1 or 2).
        """
        self.alpha = epsilon
        probs = self.model(state)
        # choose an action by its probability (to add exploration)
        action = Categorical(probs).sample().item()
        return action

    def __get_vt_list(self, reward_batch):
        """
        generates the vt list of future discounted rewards
        :param rewards: the rewards per each step
        :return: a Tensor with the vt list
        """
        Vt_list = []
        for t in range(len(reward_batch)):
            Vt, gamma_p = 0, 0
            for r in reward_batch[t:]:
                Vt += r * (self.gamma ** gamma_p)
                gamma_p += 1
            Vt_list.append(Vt)

        # normalize discounted rewards - Vt
        Vt_list = torch.tensor(Vt_list)
        return (Vt_list - Vt_list.mean()) / (Vt_list.std() + 1e-9)

    def __optimization_step(self, pg):
        """
        Compute optimization step using the optimizer on the network
        :param pg: the objective list
        :return: the loss
        """
        # Compute loss
        loss = -pg.mean()
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def __get_pg(self, state_batch, action_batch, Vt_list):
        """
        Gets the objective list
        :param state_batch: the states
        :param action_batch: the actions
        :param Vt_list: the calculates vt's
        :return: the objective list
        """
        pred_probs = self.model(state_batch)
        log_probs = torch.log(pred_probs.gather(dim=1, index=action_batch.unsqueeze(1)))
        # calculate entropy of the probabilities
        disc_entropy = self.alpha * Categorical(pred_probs).entropy()
        return torch.mul(log_probs.squeeze(), Vt_list) * disc_entropy

    def optimize(self, batch_size, global_step=None):
        """
        Optimize the policy using policy gradient/MC algorithm
        :param rewards: list of the rewards of the current batch
        :param log_probs: the log probabilities given for each (s,a) in this batch
        :param global_step:
        """
        loss_avg, total = 0, 0
        self.memory.batch_size = batch_size
        for transitions_batch in self.memory:
            # transform list of tuples into a tuple of lists.
            # explanation here: https://stackoverflow.com/a/19343/3343043
            batch = Transition(*zip(*transitions_batch))
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # get the discounted future rewards for each step t, noted Vt
            Vt_list = self.__get_vt_list(reward_batch)
            # calculate the objective (policy gradient) we want to maximize
            pg = self.__get_pg(state_batch, action_batch, Vt_list)
            # maximize the pg and update weights
            loss_avg += self.__optimization_step(pg)
            total += 1
        self.cur_loss = -loss_avg / total

    def get_current_loss(self):
        """
        :return: the average loss that was calculated in the last optimization step
        """
        return self.cur_loss

