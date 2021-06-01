"""DeepOBS utility functions shared among experiments."""

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
