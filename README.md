This repository contains the code and experiments for the submission *ViViT: Curvature access through the generalized Gauss-Newton's low-rank structure*.

# Reproducing the experiments

**Note:** Experiments were generated and verified to run on `python=3.7.6` with `pip==19.3.1`.

## Installation

Clone the repository and run the following commands in a virtual environment:

```bash
# main library requirements
pip install -r requirements.txt

# only for development/experiments
pip install -r requirements-dev.txt
pip install -r exp/requirements-exp.txt

# main library
pip install -e .
```

**Alternative:** If you use `conda`, you can call `make conda-env`. It will set up an environment called `vivit` and install the above dependencies. Activate it with `conda activate vivit` (remove it with `conda env remove -n vivit`).

## Running experiments

Each experiment, along with instructions, is contained in a subdirectory under `exp/`:

- [Performance evaluation](exp/exp01_performance/README.md)
- [Noise analysis](exp/exp02_noise_analysis/README.md)
- [Noisy quadratic](exp/exp03_noisy_quadratic/README.md)
