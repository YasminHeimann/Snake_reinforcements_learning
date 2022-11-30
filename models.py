######################################################
# Yasmin Heimann, hyasmin, 311546915
#
# @description A module that holds multiple NN models for RL Q learning and Policy Gradient
#
######################################################

## IMPORT of packages ##
import torch.nn as nn
import torch.nn.functional as F


class SimpleModel(nn.Module):
    # A simple linear model that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.out = nn.Linear(in_features=90, out_features=3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.
        center = center.flatten(start_dim=1)
        return self.out(center)


class DQN(nn.Module):
    # A simple linear model that depends on the 9x9 area around the snakes head, with 3 linear layers
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(in_features=10 * 9 * 9, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=3)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class DQN_C(nn.Module):
    # A CNN model that depends on the 9x9 area around the snakes head, with convolution layer
    def __init__(self):
        super(DQN_C, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=10, stride=2, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(90, 3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x= self.fc1(x.flatten(start_dim=1))
        return x


class PgModel(nn.Module):
    # A simple linear network for PG algorithm that depends on the 3x3 area around the snakes head.
    def __init__(self):
        super(PgModel, self).__init__()
        self.out = nn.Linear(in_features=90, out_features=3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        center = x[:, :, 3:6, 3:6]  # The 9 cells around the snakes head (including the head), encoded as one-hot.
        pred = center.flatten(start_dim=1)
        return F.softmax(self.out(pred), dim=1) # todo 1?


class PgConvModel(nn.Module):
    # A convolution network for PG algorithm that depends on the 9x9 area around the snakes head.
    def __init__(self):
        super(PgConvModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=10, out_channels=10, stride=2, kernel_size=4)
        self.bn1 = nn.BatchNorm2d(10)
        self.fc1 = nn.Linear(90, 3)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # nhwc -> nchw
        x = F.softmax(self.bn1(self.conv1(x)), dim=1)
        x = self.fc1(x.flatten(start_dim=1))
        return F.softmax(x, dim=1)