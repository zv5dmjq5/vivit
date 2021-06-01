"""Model definitions for experiments."""

import torch


def simple_cnn():
    """A simplistic, yet deep, ReLU convolutional neural network for MNIST."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 20, 5, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Conv2d(20, 50, 5, 1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2, 2),
        torch.nn.Flatten(),
        torch.nn.Linear(4 * 4 * 50, 500),
        torch.nn.ReLU(),
        torch.nn.Linear(500, 10),
    )
