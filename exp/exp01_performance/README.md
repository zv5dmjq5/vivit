In this experiment we explore the computational limits of our approach under
different approximation schemes to reduce cost.

**Hardware information:**

- CPU: Intel® Core™ i7-8700K CPU @ 3.70GHz × 12 (31.3 GiB)
- GPU: GeForce RTX 2080 Ti (11264 MB)

# Experiment 1: Critical batch size for GGN spectrum

We compute the mini-batch GGN's eigenvalue spectrum (in the Gram space) in
addition to a regular gradient backpropagation in PyTorch. Under different GGN
approximations (exact, MC=1-sampled), allocated samples (full batch, fraction of
batch), and parameter groupings (full net, layerwise) we ramp up the batch size
and find the critical value as maximum batch size that does not lead to
out-of-memory errors.

To reproduce this experiment:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python call_run_N_crit_evals.py`. Find the critical batch sizes under
   `results/N_crit/evals`.
3. Clean up with `bash clean.sh` to remove the results.

# Experiment 2: Critical batch size for Newton steps

We compute a damped Newton step, in addition to a regular gradient
backpropagation in PyTorch, and determine the critical batch size under
variation of GGN approximations (exact, MC=1-sampled), allocated samples (full
batch, fraction of batch), and parameter groupings (full net, layerwise).

To reproduce this experiment:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python call_run_N_crit_newton_step.py`. Find the critical batch sizes
   under `results/N_crit/newton_step`.
3. Clean up with `bash clean.sh` to remove the results.

# Experiment 3: GGN eigenvalue spectra

We compute and visualize the full eigenvalue spectrum for different
approximations of the GGN (exact, MC=1-sampled) and allocated samples (full
batch, fraction of batch).

To reproduce this experiment:

1. (Optional) Extract `results.zip` to use the original data with `unzip
   results.zip`
2. Run `python run_evals.py`. The results are stored under `results/evals`.
3. Plot `python plot_evals.py`. The results are stored under `fig/evals`.
3. Clean up with `bash clean.sh` to remove the results.
