![autoNN logo](./assets/pysenn-logo.png)

![Python](https://img.shields.io/badge/Python-3.14-FFD43B?style=for-the-badge&logo=python&logoColor=306998)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=red)

A Python package, based on PyTorch, to build self-expanding Neural Networks and train them with an optimized process.

I'm building it because **I want to make SENN easier to create and manage**. I'm starting from scratches, for create a really efficient and optimized structure.

## File Structure

```
PySENN/
├── src/
│   └── pysenn/
│       ├── __init__.py
│       ├── __main__.py
│       ├── playground/      # this directory has no __init__.py because hasn't got resources to use, only my practice with PyTorch essentials
│       │   ├── _01_tensors.py
│       │   └── _02_operations.py
│       ├── sell/
│       │   ├── __init__.py
│       │   ├── linear.py       # ExpLayer (SELL)
│       │   ├── filter.py       # FilterLayer
│       │   └── module.py       # ExpModule
│       └── utils/
│           ├── __init__.py
│           └── randomdata.py
├── assets/
├── dist/
├── pyproject.toml
├── uv.lock
└── README.md
```

### ExpLayer (SELL):

**The core layer**  
It is the core of the package: is built starting from a nn.Module (avoiding the nn.Linear limits) to have **a full-customizable structure**.
So I've built a linear layer that can expand his structure, adding new neurons and **improving automatically the neural network power**.
### FilterLayer:

**The bridge**  
A static layer would break if the previous layer grows from 8 to 12 neurons. 
So I've built **FilterLayer**, a layer that takes the variable output of a SELL and **return always the same sizes Tensor**, keeping the flow stable.

### ExpModule:

**The container**   
Groups one ExpLayer + one FilterLayer (putting a ReLU between them) inside a nn.ModuleList. **So you can implement SELL in your nn much easier**.

## Quickstart

You can add a SELL in your nn.Module in a very easy way. 

>![WARNING]
>Because of the self-expanding structure, for now **the package doesn't work with optimizers** (optimizers like ADAM have a gradients memory that crashes if you add new). The only way is **using SGD optimizer and recreating it every time you expand the neural network**. See below.

