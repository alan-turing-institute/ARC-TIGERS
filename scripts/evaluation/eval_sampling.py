import argparse
import json
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd


def get_metric_stats(
    data_dir: str, plot: bool = True
) -> dict[
    str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
]:
    """
    Compute and plot stats of metric values after a varying number of labelled samples.

    Args:
        data_dir: Directory containing the metrics files generated by
            random_sampling.py. Files should be named `metrics_*.csv` with each file
            containing metric values for a single loop of sampling. It should also
            contain a `metrics_full.json` file with the metrics calculated for the whole
            dataset. This is used to compare the metrics on the labelled subset with the
            metrics on the whole dataset.

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
    metric_stats: dict[
        str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
    ] = {}
    metric_stats["n_labels"] = np.array(repeats[0].index)
    n_repeats = len(repeats)
    n_samples = len(repeats[0])
    quantiles = [0.025, 0.25]  # will also compute 1 - quantiles
    quantile_colors = ["r", "orange"]
    for metric in metrics:
        metric_repeats = np.full((n_repeats, n_samples), np.nan)
        for idx, r in enumerate(repeats):
            metric_repeats[idx, :] = r[metric]

        metric_stats[metric] = {}
        metric_stats[metric]["mean"] = np.mean(metric_repeats, axis=0)
        metric_stats[metric]["median"] = np.median(metric_repeats, axis=0)
        metric_stats[metric]["std"] = np.std(metric_repeats, axis=0)

        for quantile in quantiles:
            metric_stats[metric][f"quantile_{quantile * 100}"] = np.quantile(
                metric_repeats, quantile, axis=0
            )
            metric_stats[metric][f"quantile_{(1 - quantile) * 100}"] = np.quantile(
                metric_repeats, 1 - quantile, axis=0
            )
        metric_stats[metric]["IQR"] = (
            metric_stats[metric]["quantile_75.0"]
            - metric_stats[metric]["quantile_25.0"]
        )

    if plot:
        plot_metrics(
            data_dir,
            metric_stats,
            full_metrics,
            quantiles,
            quantile_colors,
        )

    return metric_stats


def plot_metrics(
    save_dir: str,
    metric_stats: dict[
        str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
    ],
    full_metrics: dict[str, int | float],
    quantiles: list[float] | None = None,
    quantile_colors: list[str] | None = None,
):
    """Plot how metric stats vary with the number of labelled samples.

    Args:
        save_dir: Directory to save the plots.
        metric_stats: Dictionary containing the metric stats.
        full_metrics: Dictionary containing the metrics of the full dataset.
        quantiles: List of quantiles to plot. If None, defaults to [0.025, 0.25].
        quantile_colors: List of colors for the quantiles. If None, defaults to ["r",
            "orange"].
    """
    if quantile_colors is None:
        quantile_colors = ["r", "orange"]
    if quantiles is None:
        quantiles = [0.025, 0.25]

    for metric in metric_stats:
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
            metric_stats["n_labels"],
            metric_stats[metric]["mean"],
            label="Mean (labelled subset)",
            color="blue",
            linestyle="-",
            linewidth=2,
        )
        for quantile, color in zip(quantiles, quantile_colors, strict=True):
            plt.fill_between(
                metric_stats["n_labels"],
                metric_stats[metric][f"quantile_{quantile * 100}"],
                metric_stats[metric][f"quantile_{(1 - quantile) * 100}"],
                color=color,
                alpha=0.2,
                label=f"Quantiles ({quantile * 100} - {(1 - quantile) * 100})",
            )
        plt.ylabel(metric)
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/metrics_{metric}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sampling metrics.")
    parser.add_argument(
        "data_dir", type=str, help="Directory containing the metrics files."
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    get_metric_stats(data_dir)
