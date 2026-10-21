"""
automodel.py
Authors: Alessio Parolini
This file contains the resources to take a PyTorch nn.Module object and return a self-expanding-ready nn.Module
"""
from typing import Any

# creating the module class

from torch import nn


class LinearExpModule:

    """
    This class takes a nn.Module as input and return one, but with a correct structure for be trained with a self-expanding algorithm.
    """

    def __init__(self, pt_model: nn.Module):

        pass