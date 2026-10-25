"""
module.py
Authors: Alessio Parolini
Contains the class that groups ExpLinear and FilterLayer for an easier implementation
"""
from typing import Any

import torch

from torch import nn

from pysenn.sell import ExpLayer, FilterLayer

class ExpModule(nn.Module):

    def __init__(self, sell:ExpLayer, filter_layer:FilterLayer):

        super().__init__()

        self.structure = nn.ModuleList([sell, filter_layer])

    def expand(self, expansion: int)-> None:

        """
        Expand the whole structure
        :param expansion: a parameter to specify expansion. If None uses the ExpLinear defalut
        :return: None
        """

        expansion = expansion if expansion else None

        for module in self.structure:

            module.expand(expansion=expansion)

    def forward(self, x):

        """
        Process the x in the SELL structure
        :param x:
        :return:
        """

        for module in self.structure:

            x = module(x)

        return x
