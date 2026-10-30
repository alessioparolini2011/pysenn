"""
linear.py
Authors: Alessio Parolini
This file contains the resources to implement SELLs (Self Expanding Linear Layers) in a static neural network
"""

__all__ = ["ExpLayer"]

import torch

from torch.nn import functional as f

from torch import nn

class ExpLayer(nn.Module):

    def __init__(self, in_features:int, out_features:int, exp_rate:int = 4):

        """
        :param in_features: the number of weights for neuron in the layer - don't change
        :param out_features: the initial number of neurons - change over time
        :param exp_rate: how many neuron adds every time the layer expands itself
        """

        super().__init__()

        self.in_features = in_features

        self.out_features = out_features

        self.exp_rate = exp_rate

        self.weights = nn.Parameter(data=(torch.randn(self.out_features, self.in_features)))

        self._expansion = 0

    def expand(self, expansion: int)-> None:

        """
        Adds the new neurons, making expansion possible.
        :expansion: there is a difference between exp_rate and _expansion (for future improvement), so I add expansion. Also is a better structure for FilterLayer
        :return: None
        """

        self._expansion = expansion if expansion is not None else self.exp_rate

        new_neurons = torch.randn(self._expansion, self.in_features)

        self.weights = nn.Parameter(data=torch.cat([self.weights.data, new_neurons], dim=0))

        self.out_features += self._expansion

    def forward(self, x):

        """
        This function uses a particular logic, because the first epoch with new parameters, they have to be 0,
        because having no gradient they can increase the loss and ruin all the progress of the nn
        :param x: the input tensor
        :return: the layer output
        """

        out = f.linear(x, self.weights[:(self.out_features - self._expansion)])

        zeros = torch.zeros(x.size(0), self._expansion, device=x.device, dtype=x.dtype)

        self._expansion = 0

        return torch.cat([out, zeros], dim=1)