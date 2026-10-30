"""
demo.py
Authors: Alessio Parolini
This file provide a run-ready demonstration of the features of the project
"""

import torch.nn

from torch import nn

from torch import optim

from pysenn.sell import ExpModule


class MyModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.relu = nn.ReLU()

        self.layer1 = nn.Linear(in_features=1, out_features=5)

        self.sell = ExpModule(in_features=5, out_features=10)

        self.layer3 = nn.Linear(in_features=10, out_features=1)

    def forward(self, x):
        x = self.layer1(x)

        x = self.relu(x)

        x = self.sell(x)

        x = self.relu(x)

        x = self.layer3(x)

        return x

def main():

    model = MyModule()

    optimizer = optim.SGD(model.parameters(), lr=0.01)  # creating the optimizer

    criterion = nn.MSELoss()

    input_tensor = torch.rand(10, 1) #creating input

    output_tensor = torch.sin(input_tensor)

    print("\nWelcome to PySENN! This is a package features demonstration!\n")

    for epoch in range(500):

        optimizer.zero_grad()

        model_out = model(input_tensor)

        loss = criterion(model_out, output_tensor)

        loss.backward()

        optimizer.step()

        print(loss.item())

        if epoch % 5 == 0:

            model.sell.expand(3)

            print("\nExpansion done!\n")

            optimizer = optim.SGD(model.parameters(), lr=0.01)  # CREATING A NEW OPTIMIZER!

if __name__ == "__main__":

    main()
