"""Load saved models and evaluate noise."""

import os
from collections import defaultdict

import torch
from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from deepobs.pytorch.testproblems import cifar100_allcnnc
from shared import (
    CONFIGURATIONS,
    criterion,
    full_batch_exact,
    get_noise_savepath,
    load_checkpoint_data,
    one_group,
)

from exp.utils.deepobs import get_deepobs_architecture
from exp.utils.path import write_to_json


def get_mini_batch(problem_cls, N):
    """Draw mini batch on which noise is computed."""
    _, _, X, y = get_deepobs_architecture(problem_cls, N)

    return X, y


def compute_noise(model, loss_func, X, y, param_groups, computations):
    """Compute GGN eigenvalues, first- and second-order directional derivatives."""
    model.zero_grad()

    loss = loss_func(model(X), y)

    with backpack(
        *computations.get_extensions(param_groups),
        extension_hook=computations.get_extension_hook(
            param_groups,
            keep_gram_mat=False,
            keep_gram_evals=True,
            keep_gram_evecs=False,
            keep_gammas=True,
            keep_lambdas=True,
            keep_batch_size=False,
            keep_backpack_buffers=False,
        ),
    ), weight_jac_t_save_memory(save_memory=True):
        loss.backward()

    grad_norm = {}
    for group in param_groups:
        grad_flat = torch.cat([p.grad.flatten() for p in group["params"]])
        grad_norm[id(group)] = grad_flat.norm()

    return {
        "gram_evals": computations._gram_evals,
        "gammas": computations._gammas,
        "lambdas": computations._lambdas,
        "grad_norm": grad_norm,
    }


def run_noise(
    model,
    loss_func,
    X,
    y,
    device,
    param_groups_fn=one_group,
    computations_fn=full_batch_exact,
):
    """Run noise computation and return results."""
    model = extend(model.to(device))
    loss_func = extend(loss_func.to(device))
    X = X.to(device)
    y = y.to(device)

    param_groups = param_groups_fn(model, criterion)
    computations = computations_fn(N)

    noise = compute_noise(model, loss_func, X, y, param_groups, computations)
    noise = convert_to_numpy(noise)
    print(noise)

    return noise


def convert_to_numpy(dictionary):
    """Convert all torch tensors to numpy for JSON serialization."""
    converted = {}

    for key in dictionary.keys():
        item = dictionary[key]

        if isinstance(item, dict):
            converted_item = convert_to_numpy(item)
        elif isinstance(item, torch.Tensor):
            converted_item = item.detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown conversion for key {key} with item {item}")

        converted[key] = converted_item

    return converted


if __name__ == "__main__":
    configurations = [config() for config in CONFIGURATIONS]

    for config in configurations:
        # use the same mini-batch for eval, gamma, and lambda at checkpoints

        optimizer_cls = config["optimizer_cls"]
        problem_cls = config["problem_cls"]
        savepath = get_noise_savepath(problem_cls, optimizer_cls)

        if os.path.exists(savepath):
            print(f"[exp] File {savepath} already exists. Skipping computation.")
            continue

        torch.manual_seed(0)
        N = config["batch_size"]

        if problem_cls == cifar100_allcnnc:
            N = 64

        X, y = get_mini_batch(problem_cls, N)

        data = defaultdict(dict)

        for checkpoint in config["checkpoints"]:
            checkpoint_data = load_checkpoint_data(
                checkpoint, optimizer_cls, problem_cls
            )

            model = checkpoint_data.pop("model")
            loss_func = checkpoint_data.pop("loss_func")
            device = torch.device("cpu")

            epoch_count, batch_count = [int(count) for count in checkpoint]
            data[epoch_count][batch_count] = run_noise(model, loss_func, X, y, device)

        write_to_json(savepath, data)
