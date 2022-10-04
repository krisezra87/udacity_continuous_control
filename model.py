import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def set_limits(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """ The Policy Model """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        # Define the number of nodes in the network here.  Not changing
        # and keeps the signature simple
        net1_nodes = 128
        net2_nodes = 128

        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.net1 = nn.Linear(state_size, net1_nodes)
        self.net2 = nn.Linear(net1_nodes, net2_nodes)
        self.net3 = nn.Linear(net2_nodes, action_size)
        self.reset()

    def reset(self):
        self.net1.weight.data.uniform_(*set_limits(self.net1))
        self.net2.weight.data.uniform_(*set_limits(self.net2))
        self.net3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """ Create the policy network that maps states to actions """
        x = F.relu(self.net1(state))
        x = F.relu(self.net2(x))
        x = torch.tanh(self.net3(x))
        return x


class Critic(nn.Module):
    """ The Policy Model """

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """

        # Define the number of nodes in the network here.  Not changing
        # and keeps the signature simple
        net1_nodes = 128
        net2_nodes = 128

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.net1 = nn.Linear(state_size, net1_nodes)
        self.net2 = nn.Linear(net1_nodes+action_size, net2_nodes)
        self.net3 = nn.Linear(net2_nodes, 1)
        self.reset()

    def reset(self):
        self.net1.weight.data.uniform_(*set_limits(self.net1))
        self.net2.weight.data.uniform_(*set_limits(self.net2))
        self.net3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """ Map state-action pairs to Q values """
        x = F.relu(self.net1(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.net2(x))
        x = self.net3(x)
        return x
