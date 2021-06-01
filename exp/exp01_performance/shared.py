"""Shared functionality among sub-experiments."""

import copy
from typing import Type

from deepobs.pytorch import testproblems


def get_deepobs_architecture(
    problem_cls: Type[testproblems.testproblem.TestProblem], N
):
    """Get model, loss function, and data of DeepOBS problems."""
    problem = problem_cls(batch_size=N)
    problem.set_up()
    problem.train_init_op()

    model = copy.deepcopy(problem.net)
    loss_func = copy.deepcopy(problem.loss_function(reduction="mean"))
    X, y = problem._get_next_batch()

    return model, loss_func, X.clone(), y.clone()


def cifar10_3c3d(N):
    """Get model, loss_function, and data for CIFAR10-3c3d."""
    return get_deepobs_architecture(testproblems.cifar10_3c3d, N)


def fmnist_2c2d(N):
    """Get model, loss_function, and data for F-MNIST-2c2d."""
    return get_deepobs_architecture(testproblems.fmnist_2c2d, N)


def cifar100_allcnnc(N):
    """Get model, loss_function, and data for CIFAR100-All-CNNC."""
    return get_deepobs_architecture(testproblems.cifar100_allcnnc, N)


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


def paramwise_group(model, criterion):
    """Build parameter group containing one parameter per group."""
    return [{"params": [p], "criterion": criterion} for p in model.parameters()]


def layerwise_group(model, criterion):
    """Group weight and bias of a layer."""

    def is_weight(p):
        return p.dim() > 1

    def is_bias(p):
        return p.dim() == 1

    params = list(model.parameters())
    num_params = len(params)
    assert num_params % 2 == 0, "Number of torch parameters must be even."
    param_groups = []

    for idx in range(num_params // 2):
        p1 = params[2 * idx]
        p2 = params[2 * idx + 1]

        assert (is_weight(p1) and is_bias(p2)) or (is_weight(p2) and is_bias(p1))
        param_groups.append({"params": [p1, p2], "criterion": criterion})

    return param_groups
