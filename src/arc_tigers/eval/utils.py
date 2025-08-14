import json
import logging
import os
from glob import glob
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from arc_tigers.eval.plotting import plot_difference, plot_metrics

logger = logging.getLogger(__name__)


def get_replay_exp_results(source_model_dir, configs):
    """
    Get the replay experiment results for a given source model directory and a list of
    configurations.

    Args:
        source_model_dir: Path to the source model's results directory.
        configs: List of configurations to look for in the replay results.
    Returns:
        A dictionary containing the replay results for each configuration and the save
        directory for the figures.
    """
    replay_results = {}

    replay_results["base"] = get_metric_stats(source_model_dir, plot=False)

    # Split the path from the 6th location
    parts: list[str] = source_model_dir.split(os.sep)
    sampling_method = parts[-1]
    model = parts[5]
    eval_imbalance = parts[-2]

    for possible_config in configs:
        if model.lower() in possible_config.lower():
            dest_config = possible_config

    data_split_loc = 5
    if len(parts) > data_split_loc:
        replay_base_path = os.sep.join(
            [*parts[:data_split_loc], "replays", eval_imbalance, dest_config]
        )

    print("replay base path:", replay_base_path)

    for possible_config in configs:
        if possible_config == dest_config:
            continue
        replay_path = os.sep.join(
            [replay_base_path, f"from_{possible_config}", sampling_method, "all_seeds"]
        )
        if os.path.exists(replay_path):
            print(f"Found replay results for {possible_config} at {replay_path}")
            replay_results[possible_config] = get_metric_stats(replay_path, plot=False)

    save_dir = os.sep.join(
        [
            *parts[:data_split_loc],
            "figures",
            "resampling",
            model,
            eval_imbalance,
            sampling_method,
        ]
    )

    if sampling_method != "random":
        random_path = os.sep.join([*source_model_dir.split(os.sep)[:-1], "random"])
        if os.path.exists(random_path):
            print(f"Found random sampling results at {random_path}")
            replay_results["random_baseline"] = get_metric_stats(
                random_path, plot=False
            )

    return replay_results, save_dir


def find_model_directories(
    base_path: str,
    models: list[str],
    test_imbalance: str,
    sampling_method: str,
    plot_first: bool = False,
) -> dict[str, str]:
    """
    Find the sampling method directories for each model.

    Args:
        base_path: Base path like "outputs/reddit_dataset_12/one-vs-all/football/42_05"
        models: List of model names to search for
        test_imbalance: Test imbalance level (e.g., "05", "01", "001")
        sampling_method: Sampling method directory name (e.g., "random")

    Returns:
        Dictionary mapping model names to their sampling method directory paths
    """
    model_paths = {}

    for model in models:
        # Look for pattern:
        # base_path/model/*/eval_outputs/test_imbalance/sampling_method
        pattern = (
            f"{base_path}/{model}/*/eval_outputs/{test_imbalance}/{sampling_method}"
        )
        matches = glob(pattern)

        if matches:
            if len(matches) > 1:
                print("multiple matches found for model:", model)
                if plot_first:
                    print("Plotting first match only.")
                    sampling_method_path = matches[0]
                    model_paths[model] = sampling_method_path
                else:
                    print("Plotting all matches.")
                    for _, match in enumerate(matches):
                        config = match.split(os.sep)[-4]
                        model_paths[f"{model}_{config}"] = match
            else:
                model_paths[model] = matches[0]

        else:
            print(
                f"Warning: No eval_outputs found for model '{model}' "
                f"with test imbalance '{test_imbalance}' "
                f"and sampling method '{sampling_method}'"
            )
            print(f"  Searched pattern: {pattern}")

    return model_paths


def get_se_differences(target_data_dir: str, reference_data_dir: str):
    """
    Calculate the differences in SE values between target and reference directories.

    Args:
        target_data_dir: Directory containing target SE values.
        reference_data_dir: Directory containing reference SE values.

    Returns:
        A dictionary with the differences in SE values.
    """
    target_se_vals = get_se_values(target_data_dir)
    reference_se_vals = get_se_values(reference_data_dir)

    se_differences = {}
    for metric in target_se_vals:
        if metric in reference_se_vals:
            min_len = min(len(target_se_vals[metric]), len(reference_se_vals[metric]))
            target_vals = np.array(target_se_vals[metric])[:min_len]
            reference_vals = np.array(reference_se_vals[metric])[:min_len]
            se_differences[metric] = target_vals - reference_vals
        else:
            err_msg = f"Metric {metric} not found in reference data."
            raise ValueError(err_msg)

    return se_differences


def get_se_values(data_dir: str):
    """
    SE values for a level of imbalance, sampling strategy, and number of labels
    dict[sampling_strategy][imbalance_level][n_labels] = se_values
    using get_metric_stats
    """
    stats, _ = get_metric_stats(data_dir, plot=False)
    se_vals = {}

    for metric in stats:
        if metric == "n_labels":
            continue
        se_vals[metric] = stats[metric]["se"]

    return se_vals


def get_metric_stats(
    data_dir: str, plot: bool = True
) -> tuple[dict[str, Any | dict[str, Any]], Any]:
    """
    Compute and plot stats of metric values after a varying number of labelled samples.

    Args:
        data_dir: Directory containing the metrics files generated by sample.py. Files
            should be named `metrics_*.csv` with each file containing metric values for
            a single loop of sampling. It should also contain a `metrics_full.json` file
            with the metrics calculated for the whole dataset. This is used to compare
            the metrics on the labelled subset with the metrics on the whole dataset.

    Returns:
        A dictionary containing the stats of the metrics. The keys are the metric names
        and the values are dictionaries containing the mean, median, std, and quantiles
        of the metric values for each number of labelled samples. There is also a
        parent `n_labels` key containing the number of labelled samples after each
        metric calculation.
    """
    result_files = glob(f"{data_dir}/metrics_*.csv")
    if len(result_files) == 0:
        msg = "No metrics files found in the directory."
        raise ValueError(msg)
    repeats = [pd.read_csv(f, index_col="n") for f in result_files]
    if any(len(r) != len(repeats[0]) for r in repeats):
        msg = "Not all files have the same number of rows."
        raise ValueError(msg)
    if any(list(r.columns) != list(repeats[0].columns) for r in repeats):
        msg = "Not all files have the same columns."
        raise ValueError(msg)

    with open(f"{data_dir}/metrics_full.json") as f:
        full_metrics = json.load(f)
    if set(repeats[0].columns) != set(full_metrics.keys()):
        msg = "Metrics in full metrics file do not match those in the CSV files."
        raise ValueError(msg)

    metrics = repeats[0].columns
    stats: dict[
        str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
    ] = {}
    stats["n_labels"] = np.array(repeats[0].index)
    n_repeats = len(repeats)
    n_samples = len(repeats[0])
    quantiles = [0.025, 0.25]  # will also compute 1 - quantiles
    quantile_colors = ["r", "orange"]
    for metric in metrics:
        metric_repeats = np.full((n_repeats, n_samples), np.nan)
        for idx, r in enumerate(repeats):
            metric_repeats[idx, :] = r[metric]

        stats[metric] = {}
        stats[metric]["mean"] = np.mean(metric_repeats, axis=0)
        stats[metric]["median"] = np.median(metric_repeats, axis=0)
        stats[metric]["std"] = np.std(metric_repeats, axis=0)
        stats[metric]["se"] = (metric_repeats - full_metrics[metric]) ** 2
        stats[metric]["mse"] = np.mean(
            (metric_repeats - full_metrics[metric]) ** 2, axis=0
        )
        stats[metric]["mse_std"] = np.std(
            (metric_repeats - full_metrics[metric]) ** 2, axis=0
        )

        for quantile in quantiles:
            stats[metric][f"quantile_{quantile * 100}"] = np.quantile(
                metric_repeats, quantile, axis=0
            )
            stats[metric][f"quantile_{(1 - quantile) * 100}"] = np.quantile(
                metric_repeats, 1 - quantile, axis=0
            )
        stats[metric]["IQR"] = (
            stats[metric]["quantile_75.0"] - stats[metric]["quantile_25.0"]
        )

    if plot:
        plot_metrics(
            data_dir,
            stats,
            full_metrics,
            quantiles,
            quantile_colors,
        )
        plot_difference(
            data_dir,
            stats,
            full_metrics,
        )

    return stats, full_metrics


def load_metrics_file_to_df(metrics_file: str) -> pd.DataFrame:
    """Load a metrics CSV file."""
    try:
        return pd.read_csv(metrics_file)

    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return pd.DataFrame()


def get_class_percentages(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate class percentages from metrics data."""
    if df.empty or "n_class_0" not in df.columns or "n_class_1" not in df.columns:
        err_msg = "All DataFrames must contain 'n_class_0' and 'n_class_1' columns."
        raise ValueError(err_msg)

    # Calculate percentages
    df = df.copy()
    df["positive_class_pct"] = (df["n_class_1"] / df["n"]) * 100
    df["negative_class_pct"] = (df["n_class_0"] / df["n"]) * 100
    df["sample_size"] = df["n"]

    return df
