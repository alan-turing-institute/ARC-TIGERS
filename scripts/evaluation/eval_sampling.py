import argparse
import json
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(data_dir: str):
    """
    Compute and plot stats of metric values after a varying number of labelled samples.
    Here `metrics_*.csv` refers the metrics files generated in random_sampling.py. Where
    each represents a single loop of metrics calculated from the minimum number of
    labelled samples to the maximum number of labelled samples.
    The `metrics_full.json` file contains the metrics calculated from the whole dataset.
    This is used to compare the metrics of the labelled subset with the metrics of the
    whole dataset.

    Args:
        data_dir: Directory containing the metrics files.
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
    metric_stats = {}
    n_repeats = len(repeats)
    n_samples = len(repeats[0])
    for metric in metrics:
        metric_repeats = np.full((n_repeats, n_samples), np.nan)
        for idx, r in enumerate(repeats):
            metric_repeats[idx, :] = r[metric]

        metric_stats[metric] = {}
        metric_stats[metric]["mean"] = np.mean(metric_repeats, axis=0)
        metric_stats[metric]["median"] = np.median(metric_repeats, axis=0)
        metric_stats[metric]["std"] = np.std(metric_repeats, axis=0)

        quantiles = [0.025, 0.25]  # will also compute 1 - quantiles
        quantile_colors = ["r", "orange"]
        for quantile in quantiles:
            metric_stats[metric][f"quantile_{quantile * 100}"] = np.quantile(
                metric_repeats, quantile, axis=0
            )
            metric_stats[metric][f"quantile_{(1 - quantile) * 100}"] = np.quantile(
                metric_repeats, 1 - quantile, axis=0
            )

    for metric in metrics:
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
            repeats[0].index,
            metric_stats[metric]["mean"],
            label="Mean (labelled subset)",
            color="blue",
            linestyle="-",
            linewidth=2,
        )
        for quantile, color in zip(quantiles, quantile_colors, strict=True):
            plt.fill_between(
                repeats[0].index,
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
        plt.savefig(f"{data_dir}/metrics_{metric}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sampling metrics.")
    parser.add_argument(
        "data_dir", type=str, help="Directory containing the metrics files."
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    main(data_dir)
