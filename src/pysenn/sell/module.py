"""
module.py
Authors: Alessio Parolini
Contains the class that groups ExpLinear and FilterLayer for an easier implementation
"""

from torch import nn

from torch.nn import functional as f

from linear import ExpLayer

from filter import FilterLayer

class ExpModule(nn.Module):

    def __init__(self, sell:ExpLayer, filter_layer:FilterLayer):

        super().__init__()

        self.structure = nn.ModuleList([sell, filter_layer])

    def expand(self, expansion: int)-> None:

        """
        Expand the whole structure
        :param expansion: a parameter to specify expansion. If None uses the ExpLinear default
        :return: None
        """

        expansion = expansion if expansion else None

        for module in self.structure:

            module.expand(expansion=expansion)

    def forward(self, x):

        """
        Process the x in the SELL structure (+ ReLU)
        :param x:
        :return:
        """

        for module in self.structure:

            x = module(x)

            x = f.relu(x)

        return x
