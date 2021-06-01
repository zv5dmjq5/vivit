"""Toy problem with full control of 1st/2nd-order directional derivatives.

We want to optimize an objective function ``f(θ)`` w.r.t. θ. The optimizer uses
a local quadratic model ``φ(θ)`` of ``f`` around ``θ₀``. This model is
characterized by its local Hessian ``∇²ϕ(θ₀)``, gradient ``∇ϕ(θ₀)``, and value
``ϕ(θ₀)``. The Hessian is further decomposed into eigenvalue-eigenvector pairs
``{ λᵢ, eᵢ }``.

In practice, the local quadratic model is noisily observed. Only the gradients
``γᵢ = eᵢᵀ ∇ϕ(θ₀) / || eᵢ ||`` and curvatures ``λᵢ = eᵢᵀ ∇²ϕ(θ₀) eᵢ / || eᵢ
||²`` along the eigen-directions are subject to noise.
"""

import functools
import warnings
from math import sqrt

import torch
from torch import nn


class Objective:
    """Scalar objective function.

    Descendants must at least implement the following functions:

    - ``_forward``: Defines the function evaluation.

    Optional functions that descendants may implement:

    - ``_grad``: Defines the function's gradient evaluation.
    - ``_hess``: Defines the function's Hessian evaluation.
    """

    def __init__(self, func, grad_func=None, hess_func=None):
        self._func = func
        self._grad_func = grad_func
        self._hess_func = hess_func

    def forward(self, theta, detach=True):
        """Compute the objective value at a location.

        Performs checks before and after calling out to the internal ``_forward``
        implementation. These checks are necessary in order for the gradients and
        Hessians from ``autograd`` to have correct shapes.

        Args:
            theta (torch.Tensor): A 1d tensor containing the evaluation location.
            detach (bool, optional): Detach the output from the computation graph.
                Default: ``True``.

        Returns:
            torch.Tensor: 1d scalar tensor containing the function value.

        Raises:
            ValueError: If the input is not a 1d tensor.
            ValueError: If the output is not a scalar 1d tensor.
        """
        if theta.requires_grad is False:
            warnings.warn(
                "Input to forward pass is not differentiable."
                + " Hessian with autograd will be 0."
                + " (Ignore this if Hessians are computed manually)"
            )
        if theta.dim() != 1:
            raise ValueError(f"Input must be 1d tensor. Got {theta.dim()}d")

        output = self._func(theta)
        if detach:
            output = output.detach()

        if output.dim() != 1:
            raise ValueError(f"Output must be 1d tensor. Got {output.dim()}d")
        if output.numel() != 1:
            raise ValueError(f"Output must contain one element. Got {output.numel()}")

        return output

    def grad(self, theta, detach=True):
        """Evaluate the function's gradient at a location.

        Args:
            theta (torch.Tensor): A 1d tensor containing the evaluation location.
            detach (bool, optional): Detach the output from the computation graph.
                Default: ``True``.

        Returns:
            torch.Tensor: 1d tensor representing the gradient w.r.t. ``theta``.
        """
        if self._grad_func is None:
            grad = self.autograd_grad(theta)
        else:
            grad = self._grad_func(theta)

        if detach:
            grad = grad.detach()

        return grad

    def autograd_grad(self, theta):
        """Compute the function's gradient at a location.

        Args:
            theta (torch.Tensor): A 1d tensor containing the evaluation location.

        Returns:
            torch.Tensor: 1d tensor representing the gradient w.r.t. ``theta``.
        """
        return torch.autograd.grad(self.forward(theta, detach=False), theta)[0]

    def hess(self, theta, detach=True, spectral=False):
        """Compute the function's Hessian at a location.

        Args:
            theta (torch.Tensor): A 1d tensor containing the evaluation location.
            detach (bool, optional): Detach the output from the computation graph.
                Default: ``True``.
            spectral (bool, optional): Return the Hessian's spectral representation
                instead of the matrix representation. Default: ``False``.

        Returns:
            torch.Tensor: If ``spectral=False``. 2d matrix representation of the
                Hessian.

            (torch.Tensor, torch.Tensor): If ``spectral=True``. Spectral representation
                of the Hessian in terms of ``(evals, evecs)``.

        Raises:
            ValueError: If ``detach=False``. Keeping the Hessian's computation graph in
                memory can be costly and, for now, is forbidden.
            ValueError: If the output deviates from the documented convention.
        """
        if detach is False:
            raise ValueError("detach=False not supported.")

        if self._hess_func is None:
            result = self.autograd_hess(theta, spectral=spectral)
        else:
            result = self._hess_func(theta, spectral=spectral)

        if spectral:
            if not isinstance(result, tuple):
                raise ValueError(f"Expecting two tensors as output. Got {result}")
            else:
                if len(result) != 2:
                    raise ValueError(
                        f"Output tuple must have length 2. Got {len(result)}"
                    )
        else:
            if not result.dim() == 2:
                raise ValueError(f"Output must be a 2d tensor. Got {result.dim()}d")
            else:
                if result.shape[0] != result.shape[1]:
                    raise ValueError(
                        f"Output must be square matrix. Got {result.shape}"
                    )

        return result

    def autograd_hess(self, theta, spectral=False):
        """Compute the function's Hessian at a location.

        Args:
            theta (torch.Tensor): A 1d tensor containing the evaluation location.
            spectral (bool, optional): Return the Hessian's spectral representation
                instead of the matrix representation. Default: ``False``.

        Returns:
            torch.Tensor: If ``spectral=False``. 2d matrix representation of the
                Hessian.

            (torch.Tensor, torch.Tensor): If ``spectral=True``. Spectral representation
                of the Hessian in terms of ``(evals, evecs)``.
        """
        forward_no_detach = functools.partial(self.forward, detach=False)
        hess_mat = torch.autograd.functional.hessian(forward_no_detach, theta)

        if spectral:
            return hess_mat.symeig(eigenvectors=True)
        else:
            return hess_mat


def quadratic_taylor(f, location):
    """Compute the parameters for a quadratic model at a location.

    Args:
        f (Objective): The reference function to be Taylor-approximated.
        location (torch.Tensor): The location.

    Returns:
        dict: The coefficients describing a quadratic function (offset, directions,
            directional first- and second-order derivatives).
    """
    forward = f.forward(location)
    grad = f.grad(location)
    lambdas, evecs = f.hess(location, spectral=True)

    gammas = evecs.T @ grad

    return dict(forward=forward, gammas=gammas, lambdas=lambdas, evecs=evecs)


class ToyProblem:
    """Neural net training with controllable gradient/curvature noise."""

    def __init__(self, f, noise):
        """Store the true objective function and its observation noise.

        Args:
            f (callable or Objective): The true objective. Must map a 1d tensor to
                a 1d scalar output.
            noise (callable): The observation noise. Modifies a dictionary containing
                the local quadratic's coefficients and creates entries for the noise-
                corroborated quantities observed by an optimizer through autodiff.
        """
        self._f = f if isinstance(f, Objective) else Objective(f)
        self._noise = noise

    @staticmethod
    def make_modules(theta_init):
        """Create the architecture (neural net and loss function).

        Args:
            theta_init (torch.Tensor): 1d tensor whose values contain the model
                initialization.

        Returns:
            (torch.nn.Sequential, torch.nn.MSELoss): Model and loss function, loaded
                to the same device as ``theta_init``.
        """
        dim = theta_init.numel()
        device = theta_init.device

        linear = nn.Linear(dim, 1, bias=False)
        linear.weight.data = theta_init.reshape(linear.weight.shape)

        model = nn.Sequential(linear, nn.Flatten()).to(device)
        loss_func = nn.MSELoss(reduction="mean").to(device)

        return model, loss_func

    def make_data(self, theta, batch_size):
        """Draw a batch of labeled data to simulate optimizer observation noise.

        Let ``dim`` denote the optimization problem's dimension, and `num_directions`
        the number of eigenvectors.

        Args:
            theta (torch.Tensor): 1d tensor containing the current optimizer location.
            batch_size (int): Number of noisy samples to draw.

        Returns:
            (torch.Tensor, torch.Tensor, dict): The inputs (with shape
                ``[batch_size, num_directions, dim]``) and labels (with shape
                ``[batch_size, num_directions]``) of the mini-batch, and the info
                from the quadratic local Taylor approximation.

        """
        taylor_info = quadratic_taylor(self._f, theta)
        self._noise(taylor_info, batch_size)

        self._check_after_noise(taylor_info)

        evecs = taylor_info["evecs"]
        lambdas_samples = taylor_info["lambdas_samples"]
        gammas_samples = taylor_info["gammas_samples"]

        X = self._make_inputs(evecs, lambdas_samples)
        y = self._make_labels(evecs, lambdas_samples, gammas_samples, theta)

        return X, y, taylor_info

    @staticmethod
    def _check_after_noise(taylor_info):
        """Check that noise created necessary entries and holds valid numbers.

        Raises:
            ValueError: If application of the noise does not create the expected
                entries in the quadratic model's coefficient dictionary.
            ValueError: If curvature samples are non-positive.
        """
        for key in ["lambdas_samples", "gammas_samples"]:
            if key not in taylor_info.keys():
                raise ValueError(f"Noise did not create entry '{key}'")

        lambdas_samples = taylor_info["lambdas_samples"]

        if not lambdas_samples.greater(0).all():
            raise ValueError(
                f"Non-positive curvature samples detected: {lambdas_samples}"
            )

    @staticmethod
    def _make_inputs(evecs, lambdas_samples):
        """Generate inputs for the neural network.

        Let ``dim`` denote the optimization problem's dimension, and `num_directions`
        the number of eigenvectors.

        Args:
            evecs (torch.Tensor): 2d tensor of shape ``[dim, num_directions]``,
                containing the normalized eigenvalues column-wise.
            lambdas_samples (torch.Tensor): 2d tensor of shape
                ``[batch_size, num_directions]`` containing the ``λ[n, d]``.

        Returns:
            torch.Tensor: 3d tensor of shape ``[batch_size, num_directions, dim]``
                with the input to the neural net.
        """
        dim = evecs.shape[0]

        return sqrt(dim / 2) * torch.einsum("nd,id->ndi", lambdas_samples.sqrt(), evecs)

    @staticmethod
    def _make_labels(evecs, lambdas_samples, gammas_samples, theta):
        """Generate targets for the neural network.

        Let ``dim`` denote the optimization problem's dimension, and `num_directions`
        the number of eigenvectors.

        Args:
            evecs (torch.Tensor): 2d tensor of shape ``[dim, num_directions]``,
                containing the normalized eigenvalues column-wise.
            lambdas_samples (torch.Tensor): 2d tensor of shape
                ``[batch_size, num_directions]`` containing the ``λ[n, d]``.
            gammas_samples (torch.Tensor): 2d tensor of shape
                ``[batch_size, num_directions]`` containing the ``γ[n, d]``.
            theta (torch.Tensor): 1d tensor of shape ``[dim]`` containing the current
                location.

        Returns:
            torch.Tensor: 2d tensor of shape ``[batch_size, num_directions]`` with the
                labels for the neural net.
        """
        dim = evecs.shape[0]
        lambdas_samples_sqrt = lambdas_samples.sqrt()

        result = (
            torch.einsum("nd,id,i->nd", lambdas_samples_sqrt, evecs, theta.detach())
            - gammas_samples / lambdas_samples_sqrt
        )

        return sqrt(dim / 2) * result


class Quadratic:
    """Base class for a quadratic function. ``f(θ) = ¹/₂ θᵀ A θ + bᵀ θ + c.``

    Use ``autograd`` to compute gradients and Hessians.
    """

    def __init__(self, A, b, c):
        """Store coefficients of the quadratic function.

        Args:
            A (torch.Tensor): 2d symmetric matrix of dimension ``D``.
            b (torch.Tensor): 1d tensor containing ``D`` elements.
            c (torch.Tensor): Tensor containing a single number.
        """
        self._check(A, b, c)

        self._A = A
        self._b = b.reshape(-1, 1)
        self._c = c.reshape(1)

        self._evals, self._evecs = self._A.symeig(eigenvectors=True)

    @staticmethod
    def _check(A, b, c):
        """Perform dimension checks of coefficients."""
        assert A.dim() == 2
        assert A.shape[0] == A.shape[1]
        assert b.numel() == A.shape[0]
        assert c.numel() == 1

    def func(self, theta):
        """Evaluate the quadratic function."""
        theta = theta.reshape(-1, 1)
        output = 0.5 * theta.T @ self._A @ theta + self._b.T @ theta + self._c

        return output.flatten()

    def grad_func(self, theta):
        """Compute gradient manually."""
        theta = theta.reshape(-1, 1).detach()

        return (self._A.T @ theta + self._b).flatten()

    def hess_func(self, theta, spectral=False):
        """Compute Hessian manually."""
        if spectral is False:
            return self._A
        else:
            return self._evals, self._evecs
