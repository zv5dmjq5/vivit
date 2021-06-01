
In this experiment, we compare different optimizers on a noisy quadratic loss function.
In particular, we compare SGD to our second order optimizer with constant and 
directional Bootstrap damping. 

To reproduce the results:

1.  Use the original data by extracting `results.zip` with `unzip results.zip` or 
    rerun the experiment with `python run.py`
3.  Plot the results: `python plot.py`. Find the images in `results/scenario_name/`
4.  Clean up or start over: `bash clean.sh` (removes the entire `results` folder)
