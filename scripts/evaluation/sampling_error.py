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
            mse = metric_stats[experiment_name][metric]["mse"]
            plt.plot(
                metric_stats[experiment_name]["n_labels"][1:],
                mse[1:],
                label=experiment_name,
                linestyle="-",
                linewidth=2,
            )

        plt.ylabel(f"Difference to full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mse_{metric}.png", dpi=300)
        plt.close()


def plot_improvement(
    save_dir: str,
    metric_stats: dict[
        str,
        dict[
            str,
            npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]],
        ],
    ],
):
    experiment_names = list(metric_stats.keys())
    metrics = metric_stats[experiment_names[0]].keys()
    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue
        plt.axhline(y=1, color="k", linestyle="--", alpha=0.2)
        plt.gca().set_prop_cycle(None)  # Reset color cycle
        plt.gca().set_prop_cycle(
            plt.cycler(color=plt.rcParams["axes.prop_cycle"].by_key()["color"][1:])
        )
        base_mse = metric_stats[experiment_names[0]][metric]["mse"][1:]
        for experiment_name in experiment_names:
            mse = metric_stats[experiment_name][metric]["mse"]
            plt.plot(
                metric_stats[experiment_name]["n_labels"][1:],
                base_mse / mse[1:],
                label=experiment_name,
                linestyle="-",
                linewidth=2,
            )

        plt.ylabel(f"Improvement on Random Sampling: {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/improvement_{metric}.png", dpi=300)
        plt.close()


def main(directory, experiment_names, combination_name):
    save_dir = f"{directory}/figures/{combination_name}/"
    os.makedirs(save_dir, exist_ok=False)
    full_metrics = {}
    metric_stats = {}
    for experiment_name in experiment_names:
        metric_stats[experiment_name], full_metrics[experiment_name] = get_metric_stats(
            f"{directory}/eval_outputs/{experiment_name}", plot=False
        )
    plot_sampling_error(
        save_dir=save_dir,
        metric_stats=metric_stats,
    )

    plot_improvement(
        save_dir=save_dir,
        metric_stats=metric_stats,
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
        "save_folder",
        help=(
            "Name of the combination of experiments, e.g. 'reddit_balanced',"
            " under which the figures will be saved."
        ),
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Directories within /eval_outputs/ containing .csv metric files",
    )
    args = parser.parse_args()
    main(args.directory, args.experiments, args.save_folder)
