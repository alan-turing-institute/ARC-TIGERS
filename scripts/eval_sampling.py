import json
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(data_dir: str):
    """
    Compute and plot stats of metric values after a varying number of labelled samples.

    Args:
        data_dir: Directory containing the metrics files.
    """
    result_files = glob(f"{data_dir}/metrics_*.csv")
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
        metric_stats[metric]["std"] = np.std(metric_repeats, axis=0)
        metric_stats[metric]["quantile_25"] = np.quantile(metric_repeats, 0.25, axis=0)
        metric_stats[metric]["quantile_75"] = np.quantile(metric_repeats, 0.75, axis=0)
        metric_stats[metric]["quantile_2.5"] = np.quantile(
            metric_repeats, 0.025, axis=0
        )
        metric_stats[metric]["quantile_97.5"] = np.quantile(
            metric_repeats, 0.975, axis=0
        )

    for metric in metrics:
        plt.axhline(
            y=full_metrics[metric],
            color="k",
            linestyle="--",
            label="estimate (whole dataset)",
        )

        plt.plot(
            repeats[0].index,
            metric_stats[metric]["mean"],
            label="estimate (labelled subset)",
            color="blue",
            linestyle="-",
            linewidth=2,
        )
        plt.plot(
            repeats[0].index,
            metric_stats[metric]["quantile_25"],
            label="Quantiles (25-75)",
            color="orange",
            linestyle="-",
            linewidth=1,
        )
        plt.plot(
            repeats[0].index,
            metric_stats[metric]["quantile_75"],
            color="orange",
            linestyle="-",
            linewidth=1,
        )
        plt.plot(
            repeats[0].index,
            metric_stats[metric]["quantile_2.5"],
            label="Quantiles (2.5-97.5)",
            color="r",
            linestyle="-",
            linewidth=1,
        )
        plt.plot(
            repeats[0].index,
            metric_stats[metric]["quantile_97.5"],
            color="r",
            linestyle="-",
            linewidth=1,
        )
        plt.ylabel(metric)
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{data_dir}/metrics_{metric}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    data_dir = "data/random_sampling"
    main(data_dir)
