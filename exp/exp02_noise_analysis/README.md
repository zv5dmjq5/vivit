In this experiment we monitor the gradient and curvature noise while training
neural networks. The architectures are borrowed from the DeepOBS library, and we
use the hyperparameter choices of optimizer baselines from the BackPACK paper.

The procedure consists of the following steps:

1. Train a neural network. Save it at specified checkpoints.
2. For every check-pointed network, compute the distributions of first- and
   second-order directional derivatives along the non-trivial GGN directions
   over the samples of a fixed mini-batch. Save the results.
3. Visualize the noise distributions over training.

If you wish to remove all results at any step, or after reproducing the results,
you can invoke `bash clean.sh` to remove all results.

If you wish to skip the computationally expensive steps, you can skip straight
to the plotting step.

# Step 1: Neural network training

We train different architectures on different data sets with different
optimizers. The models are saved during training at specific checkpoints where
the noise will later be evaluated on.

To reproduce this step:

1. Run `python run_training.py`. The neural nets will be check-pointed under
   `results/training`.

# Step 2: Evaluate noise

For every training trajectory produced by step 1, load the model and loss
function, then evaluate the GGN spectrum, as well as first- and second-order
directional derivatives along its nontrivial directions. Store the result.

To reproduce this step:

1. Run `python run_noise.py`. The results will be saved in `results/noise`.

# Step 3: Extract metrics

From the computed noise information, extract the metrics that will then be
visualized. Save the result.

To reproduce this step:

1. Run `python run_extract.py`. Results will be stored in `results/extract`.

# Step 4: Visualize metrics

Finally create figures (saved to the `fig/` directory).

To reproduce this step:

1. (Optional) Extract `results.zip` to use the original data with `unzip
    results.zip`
2. Run the following plotting scripts (or simply call `bash plot.sh` instead):
   - `python plot_loss.py`: Plot train and test loss. Results are saved to
     `fig/loss/`.
   - `python plot_accuracy.py`: Plot train and test accuracy. Results are saved
     to `fig/accuracy/`.
   - `python plot_lambdas_gammas_mean.py`: Plot expected first- and second-order
     derivatives along the nontrivial directions. Results are saved to
     `fig/lambdas_gammas_mean/`.
   - `python plot_gammas_snr.py`: Plot signal-to-noise-ratio of first-order
     derivatives along the nontrivial eigenvectors over their associated
     eigenvalues. Results are saved to `fig/gammas_snr/`.
   - `python plot_lambdas_snr.py`: Plot signal-to-noise-ratio of second-order
     derivatives along the nontrivial eigenvectors over their associated
     eigenvalues. Results are saved to `fig/lambdas_snr/`.
   - `python plot_lambdas_snr_gammas_snr.py`: Plot signal-to-noise-ratio of
     first-order derivatives versus signal-to-noise-ratio of second-order
     derivatives along the nontrivial eigenvectors. Results are saved to
     `fig/lambdas_snr_gammas_snr/`.
   - `python plot_gradient_overlap.py`: Plot gradient overlap with nontrivial
     eigenvectors over their associated eigenvalues. Results are saved to
     `fig/gradient_curvature_overlap/`.
   - `python plot_gradient_overlap_hist.py`: Plot histogramed gradient overlaps
     with nontrivial eigenvectors over their associated eigenvalues. Results are
     saved to `fig/gradient_curvature_overlap_hist/`.
