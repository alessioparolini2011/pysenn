"""
filter.py
Authors: Alessio Parolini
This module contains the class for the filter Linear Layer to adapt a SELL to every NN
"""
import torch

from torch import nn

import torch.nn.functional as f

class FilterLayer(nn.Module):

    def __init__(self, in_features:int, out_features:int):

        super().__init__()

        self.out_features = out_features

        self.in_features = in_features

        self.weights = nn.Parameter(data=torch.randn(self.out_features, self.in_features))

        self._new_parameters = 0

    def update(self, expansion:int)-> None:

        """
        Adds the new weights in each layer neuron, making expansion possible.
        :param expansion: how many weights you want to add
        :return: None
        """

        self._new_parameters = expansion

        new_weights = torch.randn(self.out_features, expansion)

        self.weights = nn.Parameter(data=torch.cat([self.weights.data, new_weights], dim=1))

    def forward(self, x):

        """
        This function uses a particular logic, because the first epoch with new parameters, they have to be 0,
        because having no gradient they can increase the loss and ruin all the progress of the nn
        :param x: the input tensor
        :return: the layer output
        """

        out = f.linear(x, self.weights[:, :(self.weights.shape[1] - self._new_parameters)])

        zeros = torch.zeros(self.out_features, self._new_parameters)

        self.in_features += self._new_parameters

        self._new_parameters = 0

        return torch.Tensor(torch.cat([out, zeros], dim=1))

















