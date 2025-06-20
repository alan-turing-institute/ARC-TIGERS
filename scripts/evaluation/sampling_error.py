import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd

from arc_tigers.eval.utils import get_metric_stats


def load_loss_from_metrics(directory, loss_column="loss"):
    # Find all metrics_*.csv files
    files = sorted(glob(os.path.join(directory, "metrics_*.csv")))
    if not files:
        err_msg = f"No metrics_*.csv files found in {directory}"
        raise FileNotFoundError(err_msg)
    losses = []
    n_labels = []
    for f in files:
        df = pd.read_csv(f, index_col="n")
        if loss_column not in df.columns:
            err_msg = f"Column '{loss_column}' not found in {f}"
            raise ValueError(err_msg)
        losses.append(df[loss_column].values)
        n_labels = df.index.values  # Should be the same for all files
    losses = np.array(losses)  # shape: (n_repeats, n_steps)
    return n_labels, losses


def compute_mse(pred, target):
    return np.mean((pred - target) ** 2)


def plot_sampling_error(
    save_dir: str,
    metric_stats: dict[
        str,
        dict[
            str,
            npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]],
        ],
    ],
    full_metrics: dict[str, int | float],
):
    experiment_names = list(metric_stats.keys())
    metrics = metric_stats[experiment_names[0]].keys()
    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue
        plt.axhline(y=0, color="k", linestyle="--", alpha=0.2)
        for experiment_name in experiment_names:
            diff = (
                metric_stats[experiment_name][metric]["mean"]
                - full_metrics[experiment_name][metric]
            )
            diff_min = (
                metric_stats[experiment_name][metric]["mean"]
                - metric_stats[experiment_name][metric]["std"]
            ) - full_metrics[experiment_name][metric]
            diff_max = (
                metric_stats[experiment_name][metric]["mean"]
                + metric_stats[experiment_name][metric]["std"]
            ) - full_metrics[experiment_name][metric]

            plt.plot(
                metric_stats[experiment_name]["n_labels"],
                diff,
                label=experiment_name,
                linestyle="-",
                linewidth=2,
            )
            plt.fill_between(
                metric_stats[experiment_name]["n_labels"],
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


def main(directory, experiment_names):
    save_dir = f"{directory}/figures/"
    os.makedirs(save_dir, exist_ok=True)
    full_metrics = {}
    metric_stats = {}
    for experiment_name in experiment_names:
        metric_stats[experiment_name], full_metrics[experiment_name] = get_metric_stats(
            f"{directory}/eval_outputs/reddit_balanced/{experiment_name}", plot=False
        )
    plot_sampling_error(
        save_dir=save_dir,
        metric_stats=metric_stats,
        full_metrics=full_metrics,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MSE as number of samples increases."
    )
    parser.add_argument(
        "directory",
        help="Directories with /eval_outputs/",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Directories within /eval_outputs/ containing .csv metric files",
    )
    args = parser.parse_args()
    main(args.directory, args.experiments)
