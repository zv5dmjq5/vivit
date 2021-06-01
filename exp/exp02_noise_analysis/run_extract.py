"""Extrac metrics from noise for plots."""

import os

import numpy
from shared import CONFIGURATIONS, get_extract_savepath, get_noise_savepath

from exp.utils.path import read_from_json, write_to_json


def compute_snr(array: numpy.ndarray, double=True):
    """Compute the Signal-to-noise-ratio (SNR) of a batched array."""
    if double:
        array = array.astype(numpy.float64)

    def expectation(array):
        N_axis = 0
        return numpy.mean(array, axis=N_axis)

    mean = expectation(array)
    var = expectation((array - mean) ** 2)

    return mean ** 2 / var


def compute_gradient_curvature_overlap(
    gammas_mean: numpy.ndarray, grad_norm: numpy.ndarray, double=True
):
    """Compute normalized overlaps of gradients with GGN eigenvectors."""
    if double:
        gammas_mean = gammas_mean.astype(numpy.float64)
        grad_norm = grad_norm.astype(numpy.float64)

    overlap = gammas_mean ** 2 / grad_norm ** 2
    trivial = 1 - numpy.sum(overlap)

    trivial = max(0.0, trivial)

    return overlap, trivial


def extract_metrics(noise: dict):
    """Extract metrics for plots from noisy directional derivatives at a checkpoint."""
    gram_evals = noise["gram_evals"]
    gammas = noise["gammas"]
    lambdas = noise["lambdas"]
    grad_norm = noise["grad_norm"]

    # check
    assert (
        gram_evals.keys() == gammas.keys() == lambdas.keys() == grad_norm.keys()
    ), "Groups don't match"

    group_ids = gram_evals.keys()
    metrics = {}

    for group_id in group_ids:
        group_metrics = {}

        group_gram_evals = gram_evals[group_id]
        group_gammas = gammas[group_id]
        group_lambdas = lambdas[group_id]
        group_grad_norm = grad_norm[group_id]

        gammas_mean = group_gammas.mean(0)
        lambdas_mean = group_lambdas.mean(0)

        group_metrics["gram_evals"] = group_gram_evals
        group_metrics["gammas_mean"] = gammas_mean
        group_metrics["lambdas_mean"] = lambdas_mean
        group_metrics["lambdas_snr"] = compute_snr(group_lambdas)
        group_metrics["gammas_snr"] = compute_snr(group_gammas)

        (
            group_gradient_curvature_overlap,
            group_gradient_trivial_overlap,
        ) = compute_gradient_curvature_overlap(gammas_mean, group_grad_norm)
        group_metrics["gradient_curvature_overlap"] = group_gradient_curvature_overlap
        group_metrics["gradient_trivial_overlap"] = group_gradient_trivial_overlap

        metrics[group_id] = group_metrics

    return metrics


if __name__ == "__main__":
    configurations = [config() for config in CONFIGURATIONS]

    for config in configurations:

        optimizer_cls = config["optimizer_cls"]
        problem_cls = config["problem_cls"]
        savepath = get_extract_savepath(problem_cls, optimizer_cls)

        if os.path.exists(savepath):
            print(f"[exp] File {savepath} already exists. Skipping computation.")
            continue

        loadpath = get_noise_savepath(problem_cls, optimizer_cls)
        noise_data = read_from_json(loadpath)

        # nesting: epoch_count, batch_count, group_id
        data = {}

        for (epoch_count, batch_count) in config["checkpoints"]:
            epoch_count, batch_count = str(epoch_count), str(batch_count)

            if epoch_count not in data.keys():
                data[epoch_count] = {}
            if batch_count not in data[epoch_count].keys():
                data[epoch_count][batch_count] = {}

            batch_count_noise = noise_data[epoch_count][batch_count]

            metrics = extract_metrics(batch_count_noise)
            for group_id, group_metrics in metrics.items():
                data[epoch_count][batch_count][group_id] = group_metrics

        write_to_json(savepath, data)
