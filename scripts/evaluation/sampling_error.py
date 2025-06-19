import argparse
import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def plot_sampling_error(directories, target_file, loss_column="loss"):
    # Load ground truth targets (should be 1D: shape (n_steps,))
    targets = np.load(target_file)
    plt.figure(figsize=(8, 6))

    for dir_path in directories:
        n_labels, losses = load_loss_from_metrics(dir_path, loss_column=loss_column)
        # losses: shape (n_repeats, n_steps)
        mean_losses = np.mean(losses, axis=0)  # shape (n_steps,)
        mses = (mean_losses - targets) ** 2  # shape (n_steps,)
        label = os.path.basename(os.path.normpath(dir_path))
        plt.plot(n_labels, mses, label=label)

    plt.xlabel("Number of Labelled Samples")
    plt.ylabel("MSE (mean loss vs ground truth)")
    plt.title("MSE vs Number of Labelled Samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot MSE as number of samples increases."
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        required=True,
        help="Directories with .npy prediction files",
    )
    parser.add_argument(
        "--target", required=True, help="Numpy file with ground truth targets"
    )
    args = parser.parse_args()
    plot_sampling_error(args.dirs, args.target)
    plot_sampling_error(args.dirs, args.target)
