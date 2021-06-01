"""This script runs SGD and DampedNewton on a noisy quadratic. The results are stored
in `results/scenario_name`, where a subfolder is created for every optimizer. This 
script also creates `scenario_info.json` in the results-folder containing all relevant
parameters. 
"""

import json
import os

import torch
from backpack import backpack, extend

from exp.toy_problem import Objective, Quadratic, ToyProblem
from vivit.optim import BaseComputations, DampedNewton
from vivit.optim.damping import BootstrapDamping2, ConstantDamping


def apply_noise(taylor_info, samples):
    """Create entries for noisy observations of ``γ`` and ``λ``. Writes to entries
    ``'gammas_samples'`` and ``'lambdas_samples'`` in the quadratic coefficient
    dictionary ``taylor_info``.

    Args:
        taylor_info (dict): The coefficients describing a quadratic fit.
        samples (int): Number of noisy observations to be generated.
    """

    # Extract true gammas and lambdas
    gammas = taylor_info["gammas"]
    lambdas = taylor_info["lambdas"]
    D = gammas.numel()

    # Sample gammas and lambdas
    gammas_samples = torch.zeros(samples, D, device=gammas.device)
    lambdas_samples = torch.zeros(samples, D, device=lambdas.device)
    for d in range(D):

        # Sample gammas from Gaussian with mean = gammas[d] and given variance
        gam = gammas[d]
        gam_var = GAMMAS_VARIANCE[d]
        if torch.is_nonzero(gam_var):
            gammas_samples[:, d] = gam + torch.sqrt(gam_var) * torch.randn(
                samples, device=gammas.device
            )
        else:
            gammas_samples[:, d] = gam * torch.ones(samples, device=gammas.device)

        # Sample lambdas from Gamma with mean = lambdas[d] and given variance
        lam = lambdas[d]
        lam_var = LAMBDAS_VARIANCE[d]
        if torch.is_nonzero(lam_var):
            G = torch.distributions.gamma.Gamma(lam ** 2 / lam_var, lam / lam_var)
            for n in range(samples):
                lambdas_samples[n, d] = G.sample()
        else:
            lambdas_samples[:, d] = lam * torch.ones(samples, device=lambdas.device)

    taylor_info["gammas_samples"] = gammas_samples
    taylor_info["lambdas_samples"] = lambdas_samples


def crit(evals, atol=1e-2):
    """Keep the non-zero eigenvalues.

    Assumes eigenvalues to be sorted in ascending order.

    Args:
        evals (torch.Tensor): Eigenvalues.
        atol (float): Cutoff value. Smaller evals will not be considered.

    Returns:
        [int]: Indices of non-zero eigenvalues.
    """
    return [idx for idx, v in enumerate(evals) if v > atol]


def run_SGD(lr, nof_steps):
    """Run SGD with learning rate ``lr`` for ``nof_steps`` steps. Return trajectory in
    parameter space and corresponding objective function values.
    """

    # Set up problem, model, loss
    problem = ToyProblem(F_OBJECTIVE, apply_noise)
    model, loss_func = problem.make_modules(THETA_INIT.clone())
    model = model.to(DEVICE)
    loss_func = loss_func.to(DEVICE)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # Save trajectory and loss values
    trajectory = torch.zeros(D, nof_steps + 1)
    f_vals = torch.zeros(nof_steps + 1)

    # Training loop
    for step in range(nof_steps + 1):

        # Zero optimizer gradient and Newton step
        optimizer.zero_grad()

        # Forward pass
        theta = list(model.parameters())[0].flatten()
        X, y, info = problem.make_data(theta, batch_size=BATCH_SIZE)
        outputs = model(X.to(DEVICE))
        loss = loss_func(outputs, y.to(DEVICE))

        # Save results
        trajectory[:, step] = theta.detach()
        f_vals[step] = F.func(theta.detach())

        # Backward pass
        loss.backward()
        optimizer.step()
    return trajectory, f_vals


def run_DampedNewton(damping, lr, nof_steps):
    """Run the damped Newton optimizer with an instance of ``ConstantDamping`` or
    ``BootstrapDamping`` from ``vivit.optim.damping``. Use the learning rate ``lr``
    and take ``nof_steps`` steps. Return trajectory in parameter space and corresponding
    objective function values.
    """

    # Set up problem, model, loss
    problem = ToyProblem(F_OBJECTIVE, apply_noise)
    model, loss_func = problem.make_modules(THETA_INIT.clone())
    model = extend(model.to(DEVICE))
    loss_func = extend(loss_func.to(DEVICE))

    # Define optimizer
    computations = BaseComputations()
    optimizer = DampedNewton(
        model.parameters(),
        damping,
        computations,
        criterion=crit,
    )

    # Save trajectory and loss values
    trajectory = torch.zeros(D, nof_steps + 1)
    f_vals = torch.zeros(nof_steps + 1)
    dampings = torch.zeros(D, nof_steps + 1)

    # Training loop
    for step in range(nof_steps + 1):

        # Zero optimizer gradient and Newton step
        optimizer.zero_grad()
        optimizer.zero_newton()

        # Forward pass
        theta = list(model.parameters())[0].flatten()
        X, y, info = problem.make_data(theta, batch_size=BATCH_SIZE)
        outputs = model(X.to(DEVICE))
        loss = loss_func(outputs, y.to(DEVICE))

        # Save results
        trajectory[:, step] = theta.detach()
        f_vals[step] = F.func(theta.detach())

        # Backward pass
        with backpack(
            *optimizer.get_extensions(),
            extension_hook=optimizer.get_extension_hook(
                keep_gram_mat=False,
                keep_gram_evals=True,
                keep_gram_evecs=False,
                keep_gammas=False,
                keep_lambdas=False,
                keep_batch_size=False,
                keep_deltas=True,
                keep_newton_step=False,
                keep_backpack_buffers=False,
            ),
        ):
            loss.backward()
        _, gram_evlas = optimizer._computations._gram_computation._gram_evals.popitem()
        if gram_evlas.numel() > D:
            print(f"Warning: gram_evlas has {gram_evlas.numel()} > D eigenvalues")
            print("   gram_evals = \n", gram_evlas)

        # Store dampings
        _, deltas = optimizer._computations._deltas.popitem()
        dampings[:, step] = deltas

        # Take step
        optimizer.step(lr=lr)
    return trajectory, f_vals, dampings


def tensor_to_list(T):
    """Convert a ``torch.Tensor`` into a list (to be able to dump it into json)"""
    return T.detach().cpu().numpy().tolist()


def save_results(scenario_name, optimizer_name, run, trajectory, f_vals, dampings=None):
    """Save results as json file"""

    # Create folder
    HERE = os.path.abspath(__file__)
    HERE_DIR = os.path.dirname(HERE)
    optimizer_path = os.path.join(HERE_DIR, "results", scenario_name, optimizer_name)
    os.makedirs(optimizer_path, exist_ok=True)

    results_dict = {
        "trajectory": tensor_to_list(trajectory),
        "f_vals": tensor_to_list(f_vals),
    }
    if dampings is not None:
        results_dict["dampings"] = tensor_to_list(dampings)

    # Save dictionary
    file_path = os.path.join(optimizer_path, f"run_{run}.json")
    with open(file_path, "w") as json_file:
        json.dump(results_dict, json_file, indent=4)


def save_scenario(scenario_name):
    """Save scenario info as json file"""

    # Create folder
    HERE = os.path.abspath(__file__)
    HERE_DIR = os.path.dirname(HERE)
    scenario_path = os.path.join(HERE_DIR, "results", scenario_name)
    os.makedirs(scenario_path, exist_ok=True)

    # Compute some additional info
    ARGMIN, _ = torch.solve(-F_B, F_A)
    MIN = F.func(ARGMIN)
    DIST_INIT = (THETA_INIT - ARGMIN.reshape(-1)).norm()
    LOSS_INIT = F.func(THETA_INIT)

    info_dict = {
        "scenario_name": scenario_name,
        "F_A": tensor_to_list(F_A),
        "F_B": tensor_to_list(F_B),
        "F_C": tensor_to_list(F_C),
        "THETA_INIT": tensor_to_list(THETA_INIT),
        "BATCH_SIZE": BATCH_SIZE,
        "NOF_STEPS": NOF_STEPS,
        "NOF_RUNS": NOF_RUNS,
        "LR_SGD": LR_SGD,
        "LR_BDN": LR_BDN,
        "LR_CDN": LR_CDN,
        "CDN_DAMPINGS": tensor_to_list(CDN_DAMPINGS),
        "BDN_DAMPING_GRID": tensor_to_list(BDN_DAMPING_GRID),
        "GAMMAS_VARIANCE": tensor_to_list(GAMMAS_VARIANCE),
        "LAMBDAS_VARIANCE": tensor_to_list(LAMBDAS_VARIANCE),
        "ARGMIN": tensor_to_list(ARGMIN.reshape(-1)),
        "MIN": MIN.cpu().item(),
        "DIST_INIT": DIST_INIT.cpu().item(),
        "LOSS_INIT": LOSS_INIT.cpu().item(),
    }

    # Save dictionary
    file_path = os.path.join(scenario_path, "scenario_info.json")
    with open(file_path, "w") as json_file:
        json.dump(info_dict, json_file, indent=4)


# --------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------

# General
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Quadratic function
D = 20
F_A = torch.diagflat(torch.Tensor([i ** 2 for i in range(1, D + 1)])).to(DEVICE)
F_B = torch.zeros(D).reshape(D, 1).to(DEVICE)
F_C = torch.Tensor([0.0]).to(DEVICE)

# Optimization
THETA_INIT = (100 * torch.ones(D)).to(DEVICE)
BATCH_SIZE = 128
NOF_STEPS = 20
NOF_RUNS = 100

# Optimizers' parameters
LR_SGD = 1e-3
LR_BDN = 1.0
LR_CDN = 1.0
CDN_DAMPINGS = torch.logspace(-4, 2, 7)
BDN_DAMPING_GRID = torch.logspace(-4, 2, 200)

# Noise
GAMMAS_VARIANCE = 5000.0 * torch.ones(D)
LAMBDAS_VARIANCE = 50.0 * torch.ones(D)

# Dependent parameters
F = Quadratic(F_A, F_B, F_C)
F_OBJECTIVE = Objective(F.func, grad_func=F.grad_func, hess_func=F.hess_func)


if __name__ == "__main__":

    SEED_VAL = 0
    torch.manual_seed(SEED_VAL)

    # Determine scenario name and store info dictionary
    scenario_name = "gradient_and_curvature_noise"
    print(f"Scenario: {scenario_name:s}")
    save_scenario(scenario_name)

    # Run optimizers
    for run in range(NOF_RUNS):

        print(f"Run {run + 1}/{NOF_RUNS}")

        # Run SGD
        optimizer_name = "SGD"
        print("   Running " + optimizer_name)
        trajectory, f_vals = run_SGD(lr=LR_SGD, nof_steps=NOF_STEPS)
        save_results(scenario_name, optimizer_name, run, trajectory, f_vals)

        # Run BDN (directional bootstrap damping)
        optimizer_name = "BDN"
        print("   Running " + optimizer_name)
        damping = BootstrapDamping2(damping_grid=BDN_DAMPING_GRID)
        trajectory, f_vals, dampings = run_DampedNewton(
            damping, lr=LR_BDN, nof_steps=NOF_STEPS
        )
        save_results(scenario_name, optimizer_name, run, trajectory, f_vals, dampings)

        # Run CDNs with different constant dampings
        for damping in CDN_DAMPINGS:
            optimizer_name = f"CDN_{damping:.1e}"
            print("   Running " + optimizer_name)
            damping = ConstantDamping(damping)
            trajectory, f_vals, _ = run_DampedNewton(
                damping, lr=LR_CDN, nof_steps=NOF_STEPS
            )
            save_results(scenario_name, optimizer_name, run, trajectory, f_vals)
