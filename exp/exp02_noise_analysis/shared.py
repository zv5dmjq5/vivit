"""Shared functionality and configurations over experiments."""

import os

import dill
import numpy
import torch
from deepobs.config import DEFAULT_TEST_PROBLEMS_SETTINGS
from deepobs.pytorch.testproblems import cifar10_3c3d, cifar100_allcnnc, fmnist_2c2d

from exp.utils.path import read_from_json
from vivit.extensions import SqrtGGNExact
from vivit.optim import GramComputations

HEREDIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINTS_OUTPUT = os.path.join(HEREDIR, "results", "training")
NOISE_OUTPUT = os.path.join(HEREDIR, "results", "noise")
EXTRACT_OUTPUT = os.path.join(HEREDIR, "results", "extract")


# IO (extract)
def get_extract_savepath(problem_cls, optimizer_cls, extension=".json"):
    """Return save path for metric extraction from noise."""
    savedir = EXTRACT_OUTPUT
    os.makedirs(savedir, exist_ok=True)

    savepath = os.path.join(
        savedir, f"{problem_cls.__name__}_{optimizer_cls.__name__}{extension}"
    )

    return savepath


# IO (noise)
def get_noise_savepath(problem_cls, optimizer_cls, extension=".json"):
    """Return save path for noise computations."""
    savedir = NOISE_OUTPUT
    os.makedirs(savedir, exist_ok=True)

    savepath = os.path.join(
        savedir, f"{problem_cls.__name__}_{optimizer_cls.__name__}{extension}"
    )

    return savepath


# IO (training)


def load_summary(problem_cls, optimizer_cls):
    """Load summary of loss/accuracy statistics from training."""
    load_dir = get_summary_savepath(problem_cls, optimizer_cls)
    return read_from_json(load_dir)


def get_summary_savepath(problem_cls, optimizer_cls, extension=".json"):
    """Return save path for training summary."""
    savedir = get_checkpoints_savedir(problem_cls, optimizer_cls)
    return os.path.join(savedir, f"metrics{extension}")


def save_checkpoint_data(data, checkpoint, optimizer_cls, problem_cls):
    """Save data at checkpoint."""
    savepath = get_checkpoint_savepath(checkpoint, optimizer_cls, problem_cls)

    print(f"Saving to {savepath}")
    # DeepOBS models have pre-forward hooks that don't serialize with pickle.
    # See https://github.com/pytorch/pytorch/issues/1148
    torch.save(data, savepath, pickle_module=dill)


def load_checkpoint_data(checkpoint, optimizer_cls, problem_cls):
    """Load checkpointed model and loss function."""
    savepath = get_checkpoint_savepath(checkpoint, optimizer_cls, problem_cls)

    print(f"Loading from {savepath}")
    # DeepOBS models have pre-forward hooks that don't serialize with pickle.
    # See https://github.com/pytorch/pytorch/issues/1148
    return torch.load(savepath, pickle_module=dill)


def get_checkpoint_savepath(checkpoint, optimizer_cls, problem_cls, extension=".pt"):
    """Return the savepath for a checkpoint."""
    epoch_count, batch_count = checkpoint
    savedir = get_checkpoints_savedir(problem_cls, optimizer_cls)
    return os.path.join(
        savedir, f"epoch_{epoch_count:05d}_batch_{batch_count:05d}{extension}"
    )


def get_checkpoints_savedir(problem_cls, optimizer_cls):
    """Return sub-directory where checkpoint data is saved."""
    savedir = os.path.join(
        CHECKPOINTS_OUTPUT, f"{problem_cls.__name__}", f"{optimizer_cls.__name__}"
    )
    os.makedirs(savedir, exist_ok=True)

    return savedir


def add_batch_size(config):
    """Add default batch size for a run configuration."""
    problem_cls = config["problem_cls"]
    config["batch_size"] = DEFAULT_TEST_PROBLEMS_SETTINGS[problem_cls.__name__][
        "batch_size"
    ]

    return config


def add_num_epochs_and_checkpoints(config):
    """Add number of epochs and checkpoints where model will be saved to config."""
    problem_cls = config["problem_cls"]
    num_epochs = DEFAULT_TEST_PROBLEMS_SETTINGS[problem_cls.__name__]["num_epochs"]
    config["num_epochs"] = num_epochs

    if "checkpoints" not in config.keys():
        config = _add_automated_checkpoints(config)

    return config


def _add_automated_checkpoints(config):
    """Add automated grid of checkpoints."""
    num_epochs = config["num_epochs"]
    num_checkpoints = 10
    detailed = 4
    epoch_checkpoints = numpy.linspace(
        detailed, num_epochs - 1, num_checkpoints, dtype=numpy.int32
    )

    detailed_epochs = list(range(detailed))
    detailed_batches = [0, 1, 10, 100]

    checkpoints = [(epoch, 0) for epoch in epoch_checkpoints] + [
        (e, b) for e in detailed_epochs for b in detailed_batches
    ]
    config["checkpoints"] = checkpoints

    return config


def process(config):
    """Convert user-specified parameters to configuration used by the code."""
    config = add_batch_size(config)
    config = add_num_epochs_and_checkpoints(config)

    return config


# Runs (baselines from the BackPACK paper), see
# https://github.com/toiaydcdyywlhzvlob/backpack/blob/master/experiments/baselines.zip
def cifar10_3c3d_sgd():
    """Set up and return configuration for SGD run on CIFAR10-3c3d."""
    config = {
        "optimizer_cls": torch.optim.SGD,
        "hyperparams": {
            "lr": {"type": float, "default": 0.00379269019073225},
            "momentum": {"type": float, "default": 0.9},
            "nesterov": {"type": bool, "default": False},
        },
        "problem_cls": cifar10_3c3d,
        # NOTE comment out to use automated checkpoints
        # These are a manual selection from the automated grid
        "checkpoints": [(0, 0), (4, 0), (67, 0)],
    }
    return process(config)


def cifar10_3c3d_adam():
    """Set up and return configuration for Adam run on CIFAR10-3c3d."""
    config = {
        "optimizer_cls": torch.optim.Adam,
        "hyperparams": {"lr": {"type": float, "default": 0.00029763514416313193}},
        "problem_cls": cifar10_3c3d,
        # NOTE comment out to use automated checkpoints
        # These are a manual selection from the automated grid
        "checkpoints": [(0, 0), (4, 0), (67, 0)],
    }
    return process(config)


def fmnist_2c2d_sgd():
    """Set up and return configuration for run on F-MNIST-2c2d."""
    config = {
        "optimizer_cls": torch.optim.SGD,
        "hyperparams": {
            "lr": {"type": float, "default": 0.02069138081114788},
            "momentum": {"type": float, "default": 0.9},
            "nesterov": {"type": bool, "default": False},
        },
        "problem_cls": fmnist_2c2d,
        # NOTE comment out to use automated checkpoints
        # These are a manual selection from the automated grid
        "checkpoints": [(0, 0), (0, 100), (56, 0)],
    }
    return process(config)


def fmnist_2c2d_adam():
    """Set up and return configuration for Adam run on F-MNIST-2c2d."""
    config = {
        "optimizer_cls": torch.optim.Adam,
        "hyperparams": {"lr": {"type": float, "default": 0.00012742749857031334}},
        "problem_cls": fmnist_2c2d,
        # NOTE comment out to use automated checkpoints
        # These are a manual selection from the automated grid
        "checkpoints": [(0, 0), (0, 100), (56, 0)],
    }
    return process(config)


def cifar100_allcnnc_sgd():
    """Set up and return configuration for run on CIFAR100-All-CNNC."""
    config = {
        "optimizer_cls": torch.optim.SGD,
        "hyperparams": {
            "lr": {"type": float, "default": 0.04832930238571752},
            "momentum": {"type": float, "default": 0.9},
            "nesterov": {"type": bool, "default": False},
        },
        "problem_cls": cifar100_allcnnc,
        # NOTE comment out to use automated checkpoints
        # These are a manual selection from the automated grid
        "checkpoints": [(0, 0), (4, 0), (310, 0)],
    }
    return process(config)


def cifar100_allcnnc_adam():
    """Set up and return configuration for Adam run on CIFAR100-All-CNNC."""
    config = {
        "optimizer_cls": torch.optim.Adam,
        "hyperparams": {"lr": {"type": float, "default": 0.0006951927961775605}},
        "problem_cls": cifar100_allcnnc,
        # NOTE comment out to use automated checkpoints
        # These are a manual selection from the automated grid
        "checkpoints": [(0, 0), (4, 0), (310, 0)],
    }
    return process(config)


CONFIGURATIONS = [
    cifar10_3c3d_sgd,
    cifar10_3c3d_adam,
    fmnist_2c2d_sgd,
    fmnist_2c2d_adam,
    cifar100_allcnnc_adam,
    cifar100_allcnnc_sgd,
]

# computation

eigenvalue_cutoff = 1e-4


def criterion(evals, must_exceed=eigenvalue_cutoff):
    """Filter out eigenvalues close to zero.

    Args:
        evals (torch.Tensor): Eigenvalues.
        must_exceed (float, optional): Minimum value for eigenvalue to be kept.

        Returns:
            [int]: Indices of non-zero eigenvalues.
    """
    return [idx for idx, ev in enumerate(evals) if ev > must_exceed]


def one_group(model, criterion):
    """Build parameter group containing all parameters in one group."""
    return [{"params": list(model.parameters()), "criterion": criterion}]


def full_batch_exact(N):
    """Build computations to use the full batch and exact GGN."""
    return GramComputations(
        extension_cls_directions=SqrtGGNExact,
        extension_cls_second=SqrtGGNExact,
        compute_gammas=True,
        compute_lambdas=True,
        subsampling_directions=None,
        subsampling_second=None,
        verbose=False,
    )
