"""Compare GGN top eigenvalue computation with the Gram matrix (ViViT) and power
iteration (PyHessian) on CIFAR-100 ALL-CNN-C architecture."""

import random
import time
import warnings

import torch
from backpack import backpack, extend
from backpack.core.derivatives.convnd import weight_jac_t_save_memory
from backpack.hessianfree.ggnvp import ggn_vector_product

from exp.utils.deepobs import cifar100_allcnnc
from vivit import extensions
from vivit.extensions.hooks import GramSqrtGGNExact

torch.manual_seed(0)


def _ggn_evals_backpack(model, loss_func, X, y, top_n, use_lobpcg, extension, hook):
    model, loss_func = extend(model), extend(loss_func)

    outputs = model(X)
    loss = loss_func(outputs, y)

    with backpack(extension, extension_hook=hook):
        loss.backward()

    gram_matrix = hook.get_result()

    if use_lobpcg:
        evals, _ = torch.lobpcg(gram_matrix, k=top_n)
        evals, _ = torch.sort(evals)

    else:
        evals, _ = gram_matrix.symeig()
        evals = evals[-top_n:]

    return evals


def ggn_evals_exact(model, loss_func, X, y, top_n: int, use_lobpcg: bool = False):
    """Compute the GGN top eigenvalues using ViViT's Gram matrix."""
    return _ggn_evals_backpack(
        model,
        loss_func,
        X,
        y,
        top_n,
        use_lobpcg,
        extensions.SqrtGGNExact(),
        GramSqrtGGNExact(free_sqrt_ggn=True),
    )


# flake8: noqa: C901
def ggn_evals_power(model, loss_func, X, y, top_n, max_iter=100, tol=1e-3):
    """Compute the GGN top eigenvalues via power iteration."""
    outputs = model(X)
    loss = loss_func(outputs, y)

    def group_add(v1_list, v2_list, alpha=1):
        return [v1 + alpha * v2 for v1, v2 in zip(v1_list, v2_list)]

    def group_product(v1_list, v2_list):
        return sum((v1 * v2).sum() for v1, v2 in zip(v1_list, v2_list))

    def orthonormal(v_list, others):
        for other_list in others:
            v_list = group_add(
                v_list, other_list, alpha=-group_product(v_list, other_list)
            )
        normalize(v_list)
        return v_list

    def normalize(v_list):
        norm = group_product(v_list, v_list).sqrt()
        for v in v_list:
            # v /= norm
            # PyHessian: https://github.com/amirgholami/PyHessian/blob/8a61d6cea978cf610def8fe40f01c60d18a2653d/pyhessian/utils.py#L57 # noqa: B950
            v /= norm + 1e-6

    def iteration(v_list):
        new_v_list = ggn_vector_product(loss, outputs, model, v_list)
        new_eigval = group_product(v_list, new_v_list)
        normalize(new_v_list)

        return new_v_list, new_eigval

    def converged(old, new):
        """Same criterion as PyHessian."""
        # https://github.com/amirgholami/PyHessian/commit/0f7e0f63a0f132998608013351ba19955fc9d861#diff-ba06409ffbc677fe556485172e62649fe7a069631390f5a780766bff3289b06bR149-R150 # noqa: B950
        return (old - new).abs() / (old.abs() + 1e-6) < tol

    eigenvectors = []
    eigenvalues = []

    num_evals = 0
    while num_evals < top_n:

        eigval = torch.Tensor([float("inf")]).to(X.device)
        v_list = [torch.rand_like(p) for p in model.parameters()]
        normalize(v_list)

        has_converged = False

        for i in range(max_iter):
            v_list = orthonormal(v_list, eigenvectors)
            v_list, new_eigval = iteration(v_list)

            if converged(eigval, new_eigval):
                # print("Power iterations until convergence:", i)
                has_converged = True
                eigval = new_eigval
                break

            eigval = new_eigval

        if not has_converged:
            warnings.warn(f"Exceeded maximum number of {max_iter} iterations")

        eigenvalues.append(eigval)
        eigenvectors.append(v_list)
        num_evals += 1

    return torch.stack(sorted(eigenvalues))


def check_deterministic():
    """Verify that `setup` method is deterministic."""
    model1, loss_func, X1, y1 = setup()
    loss1 = loss_func(model1(X1), y1)

    model2, loss_func, X2, y2 = setup()
    loss2 = loss_func(model2(X2), y2)

    assert torch.allclose(loss1, loss2)
    assert torch.allclose(X1, X2)
    assert torch.allclose(y1, y2)
    assert all(
        torch.allclose(p1, p2)
        for p1, p2 in zip(model1.parameters(), model2.parameters())
    )


def setup():
    """Deterministically defines model, loss function, and data."""
    torch.manual_seed(0)
    random.seed(0)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 32

    model, loss_func, X, y = cifar100_allcnnc(N)
    model, loss_func, X, y = model.to(dev), loss_func.to(dev), X.to(dev), y.to(dev)

    return model, loss_func, X, y


if __name__ == "__main__":
    check_deterministic()

    # Compute top_n eigenvalues
    top_n = 1

    # via Gram matrix
    model, loss_func, X, y = setup()
    loss = loss_func(model(X), y)

    start = time.time()
    SAVE_MEMORY = True
    with weight_jac_t_save_memory(SAVE_MEMORY):
        evals_exact = ggn_evals_exact(model, loss_func, X, y, top_n)
    time_exact = time.time() - start

    # via power iteration
    model, loss_func, X, y = setup()
    loss = loss_func(model(X), y)

    start = time.time()
    evals_power = ggn_evals_power(model, loss_func, X, y, top_n)
    time_power = time.time() - start

    # Check results and print computing times
    print(f"Time (exact): {time_exact} s")
    print(f"Time (power): {time_power} s")

    print("Result (exact):", evals_exact)
    print("Result (power):", evals_power)

    if torch.allclose(evals_exact, evals_power, rtol=5e-2, atol=1e-4):
        print("Results approximately match!")
    else:
        raise ValueError("Results don't match approximately!")
