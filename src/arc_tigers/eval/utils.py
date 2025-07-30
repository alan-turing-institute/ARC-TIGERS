import json
import logging
from glob import glob
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_difference(
    save_dir: str,
    stats: dict[
        str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
    ],
    full_metrics: dict[str, int | float],
):
    for metric in stats:
        if metric == "n_labels":
            continue

        if "n_class" in metric:
            continue

        plt.axhline(y=0, color="k", linestyle="--", alpha=0.2)
        diff = stats[metric]["mean"] - full_metrics[metric]
        diff_min = (stats[metric]["mean"] - stats[metric]["std"]) - full_metrics[metric]
        diff_max = (stats[metric]["mean"] + stats[metric]["std"]) - full_metrics[metric]

        plt.plot(
            stats["n_labels"],
            diff,
            label="Mean (labelled subset)",
            linestyle="-",
            linewidth=2,
        )
        plt.fill_between(
            stats["n_labels"],
            diff_min,
            diff_max,
            linestyle="-",
            alpha=0.3,
            linewidth=2,
        )

        plt.ylabel(f"Difference to full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/differences_{metric}.png", dpi=300)
        plt.close()


def plot_metrics(
    save_dir: str,
    stats: dict[
        str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
    ],
    full_metrics: dict[str, int | float],
    quantiles: list[float] | None = None,
    quantile_colors: list[str] | None = None,
):
    """Plot how metric stats vary with the number of labelled samples.

    Args:
        save_dir: Directory to save the plots.
        stats: Dictionary containing the metric stats.
        full_metrics: Dictionary containing the metrics of the full dataset.
        quantiles: List of quantiles to plot. If None, defaults to [0.025, 0.25].
        quantile_colors: List of colors for the quantiles. If None, defaults to ["r",
            "orange"].
    """
    if quantile_colors is None:
        quantile_colors = ["r", "orange"]
    if quantiles is None:
        quantiles = [0.025, 0.25]

    limits_dict = {
        "accuracy": (0.5, 1.0),
        "loss": (0.0, 2.0),
        "n_class_0": (0, np.max(stats["n_labels"])),
        "n_class_1": (0, np.max(stats["n_labels"])),
    }

    for metric in stats:
        if metric == "n_labels":
            continue
        if "n_class" not in metric:
            # n_class on full dataset is just the full dataset size, which isn't helpful
            # to plot
            plt.axhline(
                y=full_metrics[metric],
                color="k",
                linestyle="--",
                label="Ground Truth",
            )

        plt.plot(
            stats["n_labels"],
            stats[metric]["mean"],
            label="Mean (labelled subset)",
            color="blue",
            linestyle="-",
            linewidth=2,
        )
        for quantile, color in zip(quantiles, quantile_colors, strict=True):
            plt.fill_between(
                stats["n_labels"],
                stats[metric][f"quantile_{quantile * 100}"],
                stats[metric][f"quantile_{(1 - quantile) * 100}"],
                color=color,
                alpha=0.2,
                label=f"Quantiles ({quantile * 100} - {(1 - quantile) * 100})",
            )
        y_min = limits_dict.get(metric, (-0.1, 1.1))[0]
        y_max = limits_dict.get(metric, (-0.1, 1.1))[1]
        plt.ylim(y_min, y_max)
        plt.ylabel(metric)
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/metrics_{metric}.png", dpi=300)
        plt.close()


def model_comparison_mse(
    save_dir: str,
    metric_stats: dict[
        str,
        dict[
            str,
            npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]],
        ],
    ],
):
    model_names = list(metric_stats.keys())
    metrics = list(metric_stats[model_names[0]].keys())
    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue
        plt.figure(figsize=(7, 5))
        plt.axhline(
            y=0,
            color="k",
            linestyle="--",
            alpha=0.5,
        )
        for model_name in model_names:
            mse = metric_stats[model_name][metric]["mse"]
            mse = metric_stats[model_name][metric]["mse"]
            # ignore first index as mse is very high at low label counts
            x_vals = metric_stats[model_name]["n_labels"][1:]  # type: ignore[index]
            y_vals = mse[1:]
            plt.plot(x_vals, y_vals, label=model_name, linestyle="-", linewidth=2)

        plt.ylabel(f"MSE from full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Model Comparison: {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mse/{metric}.png", dpi=300)
        plt.close()


def model_comparison_raw(
    save_dir: str,
    metric_stats: dict[
        str,
        dict[
            str,
            npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]],
        ],
    ],
    full_metrics: dict[str, dict[str, int | float]] | None = None,
):
    """Plot the raw metric values for each model."""
    model_names = list(metric_stats.keys())
    metrics = metric_stats[model_names[0]].keys()
    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue
        plt.figure(figsize=(7, 5))

        for model_name in model_names:
            mean_values = metric_stats[model_name][metric]["mean"]
            std_values = metric_stats[model_name][metric]["std"]
            x_vals = metric_stats[model_name]["n_labels"]

            plt.plot(
                x_vals,
                mean_values,
                label=model_name,
                linestyle="-",
                linewidth=2,
                markersize=4,
            )
            plt.fill_between(
                x_vals, mean_values - std_values, mean_values + std_values, alpha=0.2
            )

            # Plot baseline (full test set) for this specific model if available
            if full_metrics and model_name in full_metrics and "n_class" not in metric:
                baseline_value = full_metrics[model_name][metric]
                # Use same color as model line
                model_color = plt.gca().lines[-1].get_color()
                plt.axhline(
                    y=baseline_value,
                    color=model_color,
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1,
                )

        plt.ylabel(f"{metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Model Comparison: {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/raw/{metric}.png", dpi=300)
        plt.close()


def sampling_comparison_raw(
    save_dir: str,
    metric_stats: dict[str, dict[str, npt.NDArray | dict[str, npt.NDArray]]],
    full_metrics: dict[str, dict[str, int | float]] | None = None,
):
    """Plot raw metric values for sampling strategy comparison."""
    strategy_names = list(metric_stats.keys())
    metrics = metric_stats[strategy_names[0]].keys()

    for metric in metrics:
        if metric == "n_labels":
            continue

        plt.figure(figsize=(7, 5))

        # Plot baseline (full test set) if available
        if full_metrics and "n_class" not in metric:
            # Use the baseline from the first strategy (they should all be the same)
            baseline_value = next(iter(full_metrics.values()))[metric]
            plt.axhline(
                y=baseline_value,
                color="k",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Full Test Set",
            )

        for strategy_name in strategy_names:
            mean_values = metric_stats[strategy_name][metric]["mean"]
            std_values = metric_stats[strategy_name][metric]["std"]
            x_vals = metric_stats[strategy_name]["n_labels"]

            plt.plot(
                x_vals,
                mean_values,
                label=strategy_name,
                linestyle="-",
                linewidth=2,
                markersize=4,
            )
            plt.fill_between(
                x_vals, mean_values - std_values, mean_values + std_values, alpha=0.2
            )

        plt.ylabel(f"{metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Sampling Strategy Comparison: Raw {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/raw/{metric}.png", dpi=300)
        plt.close()


def sampling_comparison_mse(
    save_dir: str,
    metric_stats: dict[str, dict[str, npt.NDArray | dict[str, npt.NDArray]]],
):
    """Plot MSE (sampling error) for sampling strategy comparison."""
    strategy_names = list(metric_stats.keys())
    metrics = metric_stats[strategy_names[0]].keys()

    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue

        plt.figure(figsize=(7, 5))
        plt.axhline(
            y=0,
            color="k",
            linestyle="--",
            alpha=0.5,
        )

        for strategy_name in strategy_names:
            mse = metric_stats[strategy_name][metric]["mse"]
            x_vals = metric_stats[strategy_name]["n_labels"][1:]  # type: ignore[index]
            y_vals = mse[1:]

            plt.plot(x_vals, y_vals, label=strategy_name, linestyle="-", linewidth=2)

        plt.ylabel(f"MSE from full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Sampling Strategy Comparison: Sampling Error {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mse/{metric}.png", dpi=300)
        plt.close()


def sampling_comparison_improvement(
    save_dir: str,
    metric_stats: dict[str, dict[str, npt.NDArray | dict[str, npt.NDArray]]],
):
    """Plot improvement of sampling strategies relative to random baseline."""
    strategy_names = list(metric_stats.keys())
    metrics = metric_stats[strategy_names[0]].keys()

    # Check if random is available as baseline
    if "random" not in strategy_names:
        print("Warning: 'random' strategy not found, cannot compute improvements")
        return

    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue

        plt.figure(figsize=(12, 6))
        plt.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Random")

        # Use random as baseline for comparison
        baseline_mse = metric_stats["random"][metric]["mse"][1:]
        x_vals = metric_stats["random"]["n_labels"][1:]  # type: ignore[index]

        for strategy_name in strategy_names:
            mse = metric_stats[strategy_name][metric]["mse"]
            if strategy_name == "random":
                continue
            # Calculate improvement: baseline_mse / current_mse
            # Values > 1 mean improvement, < 1 mean degradation
            improvement = baseline_mse / mse[1:]
            plt.plot(
                x_vals,
                improvement,
                label=strategy_name,
                linestyle="-",
                linewidth=2,
            )

        plt.ylabel(f"Improvement over random: {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Sampling Strategy Comparison: Improvement in {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/improvement/{metric}.png", dpi=300)
        plt.close()


def find_model_directories(
    base_path: str, models: list[str], test_imbalance: str, sampling_method: str
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
            # Take the first match - this is the full path to the sampling method
            # directory
            sampling_method_path = matches[0]
            model_paths[model] = sampling_method_path
        else:
            print(
                f"Warning: No eval_outputs found for model '{model}' "
                f"with test imbalance '{test_imbalance}' "
                f"and sampling method '{sampling_method}'"
            )
            print(f"  Searched pattern: {pattern}")

    return model_paths


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
