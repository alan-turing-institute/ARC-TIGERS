## Evaluation (individual experiments)

## `eval_sampling.py`

This script analyzes and visualizes the results of sampling experiments. It uses the `get_metric_stats` function from `arc_tigers.eval.utils` to compute statistics (mean, std, quantiles, etc.) for evaluation metrics (accuracy, loss, etc.) as a function of the number of labeled samples. It also generates plots comparing the performance of sampled subsets to the full dataset.


## `sampling_error.py`

This script compares the sampling error (mean squared error, MSE) of different sampling strategies as the number of labeled samples increases. It uses metric statistics (such as MSE) computed by get_metric_stats from arc_tigers.eval.utils and generates plots to visualize the error and improvement over random sampling.


## `plot_errors.py`

Plots interquartile ranges for a set of experiments with different imbalance levels.
