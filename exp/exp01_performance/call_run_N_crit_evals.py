"""Determine critical batch size for GGN eigenvalue computations.

Call every setting in a fresh python session for independence across runs.
"""

import functools
import os

from run_N_crit_evals import __name__ as SCRIPT
from run_N_crit_evals import (
    frac_batch_exact,
    frac_batch_mc,
    full_batch_exact,
    full_batch_mc,
)
from shared import layerwise_group  # noqa: F401
from shared import cifar10_3c3d, cifar100_allcnnc, fmnist_2c2d, one_group
from shared_call import bisect, get_available_devices, run_batch_size

from exp.utils.path import write_to_json

# IO
HERE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE_DIR, "results", "N_crit", "evals")
SCRIPT = f"{SCRIPT}.py"

# Define settings
computations_cases = [
    full_batch_exact.__name__,
    full_batch_mc.__name__,
    frac_batch_exact.__name__,
    frac_batch_mc.__name__,
]
param_groups_cases = [
    one_group.__name__,
    # layerwise_group.__name__,
]
architecture_cases = [
    cifar10_3c3d.__name__,
    fmnist_2c2d.__name__,
    cifar100_allcnnc.__name__,
]
device_cases = get_available_devices(exclude_cpu=False)

# search space
N_min, N_max = 1, 32768

# find N_crit, write results to file
for architecture in architecture_cases:
    for device in device_cases:
        for param_groups in param_groups_cases:

            DATA_FILE = os.path.join(
                DATA_DIR, f"{architecture}_{device}_{param_groups}.json"
            )
            if os.path.exists(DATA_FILE):
                print(
                    "[exp10] Skipping computation. "
                    + f"Output file already exists: {DATA_FILE}"
                )
                continue
            else:
                os.makedirs(DATA_DIR, exist_ok=True)
            DATA = {}

            for computations in computations_cases:

                run_func = functools.partial(
                    run_batch_size,
                    script=SCRIPT,
                    device=device,
                    architecture=architecture,
                    param_groups=param_groups,
                    computations=computations,
                    show_full_stdout=False,
                    show_full_stderr=False,
                )
                N_crit = bisect(run_func, N_min, N_max)
                DATA[computations] = N_crit
                print(f"overflow @ N > {N_crit}")

            write_to_json(DATA_FILE, DATA)
