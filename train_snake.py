######################################################
# Yasmin Heimann, hyasmin, 311546915
#
# @description A module that learns a policy for a snake game. The possible policies are Q-Learning and MC-PG
#
######################################################

## IMPORT of packages ##
import shutil
import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import math
import argparse
import os
import sys
import matplotlib.pyplot as plt

## IMPORT of models ##
from q_policy import QPolicy
from Policy_Gradient import PolicyGradient
from snake_wrapper import SnakeWrapper
from models import SimpleModel, DQN, DQN_C, PgModel, PgConvModel


ANALYSIS = False

def create_model(network_name):
    """
    Kind-of model factory.
    Edit it to add more models.
    :param network_name: The string input from the terminal
    :return: The model
    """
    if network_name == 'simple':
        return SimpleModel()
    if network_name == 'dqn_linear':
        return DQN()
    if network_name == 'dqn_conv':
        return DQN_C()
    if network_name == 'pg':
        return PgModel()
    if network_name == 'pg_conv':
        return PgConvModel()
    else:
        raise Exception('net {} is not known'.format(network_name))


def create_policy(policy_name,
                  buffer_size, gamma, model: torch.nn.Module, writer: SummaryWriter, lr, network_name):
    """
    Policy factory from passed arguments
    :param policy_name: The string input from the terminal
    :param buffer_size: size of policy's buffer
    :param gamma: reward decay factor
    :param model: the pytorch model
    :param writer: tensorboard summary writer
    :param lr: initial learning rate
    :return: A policy object
    """
    if policy_name == 'dqn':
        return QPolicy(buffer_size, gamma, model, SnakeWrapper.action_space, writer, lr, network_name)
    if policy_name == 'pg':
        return PolicyGradient(buffer_size, gamma, model, SnakeWrapper.action_space, writer, lr, network_name)
    else:
        raise Exception('algo {} is not known'.format(policy_name))


def plot_statistics(reward_history, losses, network_name):
    """
    Plots statistic of a given training set
    :param reward_history: reward history
    :param losses: average loss history through the learning process
    :param network_name: name of the network used to leaning
    """
    batches_size = 200
    avg_rewards = []
    for i in range(int(len(reward_history) / batches_size)):
        if len(reward_history) >= ((i + 1) * batches_size):
            avg_r = sum(reward_history[i * batches_size:(i + 1) * batches_size]) / batches_size
            avg_rewards.append(avg_r)
    #print("reward history: ", reward_history)
    #print("reward avg: ", avg_rewards)
    plt.plot(avg_rewards)
    plt.title('rewards, network: ' + network_name)
    plt.show()
    plt.clf()
    #print("loss: ", losses)
    plt.title('loss average for each optimization step, network: ' + network_name)
    plt.plot(losses)
    plt.show()


def train(steps, buffer_size, opt_every,
          batch_size, lr, max_epsilon, policy_name, gamma, network_name,
          log_dir):
    """
    A function that trains an agent - snake, in a Snake game, by learning the states and best actions.
    :param steps: number of steps to learn
    :param buffer_size: buffer size
    :param opt_every: indicates after how many steps to optimize
    :param batch_size: the size to take each time we learn
    :param lr: learning rate for the model
    :param max_epsilon: the starting epsilon for exploration rate
    :param policy_name: the name of the policy to learn
    :param gamma: the discount factor of the rewards
    :param network_name: the name of the network to create
    :param log_dir: the name of the folder to log the results in
    :return: a list with the run results for optional anlysis
    """
    # create the relevant models and environment
    model = create_model(network_name)
    game = SnakeWrapper()
    writer = SummaryWriter(log_dir=log_dir)
    policy = create_policy(policy_name, buffer_size, gamma, model, writer, lr, network_name)

    state = game.reset()
    state_tensor = torch.FloatTensor(state)
    reward_history = []
    losses = []

    for step in tqdm(range(steps)):
        # epsilon exponential decay
        epsilon = max_epsilon * math.exp(-1. * step / (steps / 2))
        writer.add_scalar('training/epsilon', epsilon, step)

        # get the next action predicted
        prev_state_tensor = state_tensor
        action = policy.select_action(state_tensor, epsilon)
        state, reward = game.step(action)
        reward_history.append(reward)

        state_tensor = torch.FloatTensor(state)
        reward_tensor = torch.FloatTensor([reward])
        action_tensor = torch.LongTensor([action])

        policy.record(prev_state_tensor, action_tensor, state_tensor, reward_tensor)
        writer.add_scalar('training/reward', reward_history[-1], step)

        # optimize the agent's policy
        if step % opt_every == opt_every - 1:
            policy.optimize(batch_size=batch_size)
            loss = policy.get_current_loss()
            losses.append(loss)
            writer.add_scalar('training/batch_loss', loss, step)

    plot_statistics(reward_history, losses, network_name)
    writer.close()
    return [reward_history, losses, gamma]


def parse_args():
    """
    Arguments parser
    :return: the arguments
    """
    p = argparse.ArgumentParser()

    # tensorboard
    p.add_argument('--name', type=str, required=True, help='the name of this run')
    p.add_argument('--log_dir', type=str, required=True, help='directory for tensorboard logs (common to many runs)')

    # loop
    p.add_argument('--steps', type=int, default=10000, help='steps to train')
    p.add_argument('--opt_every', type=int, default=100, help='optimize every X steps')

    # opt
    p.add_argument('--buffer_size', type=int, default=800)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('-e', '--max_epsilon', type=float, default=0.3, help='for pg, use max_epsilon=0')
    p.add_argument('-g', '--gamma', type=float, default=.3)
    p.add_argument('-p', '--policy_name', type=str, choices=['dqn', 'pg', 'a2c'], required=True)
    p.add_argument('-n', '--network_name', type=str, choices=['simple', 'small', 'dqn_linear', 'dqn_conv',
                                                              'pg', 'pg_conv'], required=True)

    args = p.parse_args()
    return args


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")


def plot_rewards(runs, epsilon, network, fig_name, policy):
    """
    Plots the reward of multiple runs together
    :param runs: a list with the runs data
    :param epsilon: the epsilon used in the runs
    :param network: the network namr
    :param fig_name: the name to save the figure in
    :param policy: the policy used
    """
    batches_size = 100
    for run in runs:
        avg_rewards = []
        reward_history = run[0]
        for i in range(int(len(reward_history) / batches_size)):
            if len(reward_history) >= ((i + 1) * batches_size):
                avg_r = sum(reward_history[i * batches_size:(i + 1) * batches_size]) / batches_size
                avg_rewards.append(avg_r)
        # plot the graph for the current gamma
        gamma = run[2]
        label = 'gamma = ' + str(gamma)
        plt.plot(avg_rewards, label=label)
    plt.xlabel('Learning Step (Divided in ' + str(batches_size) + ')')
    plt.ylabel('Average Reward in the last ' + str(batches_size) + ' Steps')
    plt.title('Rewards for epsilon = ' + str(epsilon)+ ', Policy: ' + policy + ', Network: ' + network)
    plt.legend()
    # Display a figure.
    fig_name = fig_name + ".png"
    plt.savefig(fig_name, dpi=600)
    plt.show()
    plt.clf()


def plot_loss(runs, epsilon, network, fig_name, policy):
    """
    Plots the average loss of multiple runs together
    :param runs: a list with the runs data
    :param epsilon: the epsilon used in the runs
    :param network: the network namr
    :param fig_name: the name to save the figure in
    :param policy: the policy used
    """
    for run in runs:
        losses = run[1]
        gamma = run[2]
        label = 'gamma = ' + str(gamma)
        plt.plot(losses, label=label)
    plt.xlabel('Optimization Step')
    plt.ylabel('Average Loss in the Optimization Step')
    plt.title('Loss for epsilon = ' + str(epsilon) + ', Policy: ' + policy + ', Network: ' + network)
    plt.legend()
    # Display a figure.
    fig_name = fig_name + ".png"
    plt.savefig(fig_name, dpi=600)
    plt.show()
    plt.clf()


def run_test(network, epsilon, gamma, policy, steps):
    """
    Runs a test over multiple runs, and plots statistics
    :param network: the network name
    :param epsilon: the epsilon list to use for the runs
    :param gamma: the gamma list to use
    :param policy: the policy to use
    :param steps: number of steps for learning
    """
    for n in network:
        for e in epsilon:
            e_g_data = []
            for g in gamma:
                name = policy + "_" + n + '_e' + str(e)[2:] + '_g' + str(g)[2:]
                g_data = train(steps=steps, buffer_size=800, opt_every=100,
                      batch_size=32, lr=0.01, max_epsilon=e, policy_name=policy, gamma=g, network_name=n,
                      log_dir='./log_dir\\' + name)
                e_g_data.append(g_data)
            plot_name_r = policy + "_" + n + '_e' + str(e)[2:] + '_rewards'
            plot_name_l = policy + "_" + n + '_e' + str(e)[2:] + '_loss'
            plot_rewards(e_g_data, e, n, plot_name_r, policy)
            plot_loss(e_g_data, e, n, plot_name_l, policy)


def test_parameters():
    """
    Tests multiple parameters and plots its statistics.
    """
    epsilon = [0.05, 0.1, 0.3, 0.5, 0.9]
    gamma = [0.05, 0.1, 0.3, 0.5, 0.9]
    # dqn
    dqn_network = ['dqn_conv'] #['dqn_linear', 'dqn_conv', 'simple']
    run_test(dqn_network, epsilon, gamma, 'dqn', steps=10000)
    # pg
    pg_network = ['pg', 'pg_conv']
    run_test(pg_network, epsilon, gamma, 'pg', steps=20000)


if __name__ == '__main__':
    if ANALYSIS:
        test_parameters()
    else:
        args = parse_args()
        args.log_dir = os.path.join(args.log_dir, args.name)

        if os.path.exists(args.log_dir):
            shutil.rmtree(args.log_dir)
            if query_yes_no('You already have a run called {}, override?'.format(args.name)):
                shutil.rmtree(args.log_dir)
            else:
                exit(0)

        del args.__dict__['name']
        train(**args.__dict__)


