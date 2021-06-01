"""Shared functionality among eigenvalue sub-experiments."""

from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from shared import criterion

from vivit.extensions import SqrtGGNExact, SqrtGGNMC
from vivit.optim import GramComputations


def compute_ggn_gram_evals(model, loss_func, X, y, param_groups, computations):
    """Compute the GGN eigenvalues."""

    loss = loss_func(model(X), y)

    with backpack(
        *computations.get_extensions(param_groups),
        extension_hook=computations.get_extension_hook(
            param_groups,
            keep_gram_mat=False,
            keep_gram_evals=True,
            keep_gram_evecs=False,
            keep_gammas=False,
            keep_lambdas=False,
            keep_batch_size=False,
            keep_backpack_buffers=False,
        ),
    ), weight_jac_t_save_memory(save_memory=True):
        loss.backward()

    return computations._gram_evals


def run_ggn_gram_evals(architecture_fn, param_groups_fn, computations_fn, N, device):
    """Build model, data, and run GGN spectral computations."""
    model, loss_func, X, y = architecture_fn(N)
    model = extend(model.to(device))
    loss_func = extend(loss_func.to(device))
    X = X.to(device)
    y = y.to(device)

    param_groups = param_groups_fn(model, criterion)
    computations = computations_fn(N)

    gram_evals = compute_ggn_gram_evals(
        model, loss_func, X, y, param_groups, computations
    )
    print(gram_evals)
    return gram_evals


def compute_num_trainable_params(architecture_fn):
    """Evaluate the number of trainable model parameters."""
    N_dummy = 1
    model, _, _, _ = architecture_fn(N_dummy)

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def full_batch_exact(N):
    """Build computations to use the full batch and exact GGN."""
    return GramComputations(
        extension_cls_directions=SqrtGGNExact,
        extension_cls_second=SqrtGGNExact,
        compute_gammas=False,
        compute_lambdas=False,
        subsampling_directions=None,
        subsampling_second=None,
        verbose=False,
    )


def full_batch_mc(N):
    """Build computations to use the full batch and GGN-MC."""
    return GramComputations(
        extension_cls_directions=SqrtGGNMC,
        extension_cls_second=SqrtGGNMC,
        compute_gammas=False,
        compute_lambdas=False,
        subsampling_directions=None,
        subsampling_second=None,
        verbose=False,
    )


def frac_batch_exact(N):
    """Build computations to use a fraction of the mini-batch and the exact GGN."""
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    frac = 8
    return GramComputations(
        extension_cls_directions=SqrtGGNExact,
        extension_cls_second=SqrtGGNExact,
        compute_gammas=False,
        compute_lambdas=False,
        subsampling_directions=list(range(max(N // frac, 1))),
        subsampling_second=list(range(max(N // frac, 1))),
        verbose=False,
    )


def frac_batch_mc(N):
    """Build computations to use a fraction of the mini-batch and the GGN-MC."""
    # Fig. 3 of https://arxiv.org/pdf/1712.07296.pdf uses 1/8 and 1/4 of the samples
    frac = 8
    return GramComputations(
        extension_cls_directions=SqrtGGNMC,
        extension_cls_second=SqrtGGNMC,
        compute_gammas=False,
        compute_lambdas=False,
        subsampling_directions=list(range(max(N // frac, 1))),
        subsampling_second=list(range(max(N // frac, 1))),
        verbose=False,
    )
