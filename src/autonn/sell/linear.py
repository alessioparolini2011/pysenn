"""
linear.py
Authors: Alessio Parolini
This file contains the resources to implement a SENN algorithm in PyTorch nn.Linear layers
"""

__all__ = ["LinearLyr"]

from torch import nn

class LinearLyr:

    """
    This class groups and handles the nn.Linear layers the user choose to add ad the SENN
    """

    def __init__(self, fisher: str = "diagonal"):

        self.layers = []

        variants = ["diagonal", "kfac"]

        if fisher not in variants:

            raise ValueError(
                f"argument \"fisher\" has to be one of {", ".join(variants)}"
            )

    def __call__(self, layer:nn.Linear, add_filter_layer:bool = True)-> nn.Sequential:

        if not isinstance(layer, nn.Linear):

            raise TypeError(
                f"""argument \"layer\" has to be a nn.Linear instance
                                                             ^^^^^^^^
                """
            )

        self.layers.append(layer)

        if add_filter_layer:

            filter_layer = nn.Linear(in_features=layer.out_features, out_features=layer.out_features)

            return nn.Sequential(layer, filter_layer)

        else:

            return nn.Sequential(layer)

    def show(self)-> None:

        for idx, layer in enumerate(self.layers):

            print(f"[{idx}] -> {layer}")