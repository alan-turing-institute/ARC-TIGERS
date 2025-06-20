import argparse

import matplotlib.pyplot as plt

from arc_tigers.eval.utils import get_metric_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Plot inter-quartile ranges from a range of experiments with different "
            "imbalances."
        )
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Parent directory containing experiment sub-directories",
    )
    parser.add_argument(
        "experiment_prefix",
        type=str,
        help=(
            "Prefix for the each experiment directory. Results files for each "
            "experiment (level of imbalance) are expected to be in "
            "be in <data_dir>/<experiment_prefix>_<imbalance>"
        ),
    )
    parser.add_argument(
        "--imbalances",
        type=float,
        nargs="+",
        default=(0.01, 0.1, 0.25, 0.5),
        help="List of imbalances to plot (default: 0.01, 0.1, 0.25, 0.5).",
    )
    parser.add_argument(
        "--measure",
        type=str,
        default="MSE",
        help="Uncertainty measure to plot (default: MSE).",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    imbalance_stats = {}
    for imb in args.imbalances:
        imb_dir = (
            f"{data_dir}/imbalanced_random_sampling_outputs_{str(imb).replace('.', '')}"
        )
        imbalance_stats[imb] = get_metric_stats(imb_dir, plot=False)

    for metric in imbalance_stats[args.imbalances[0]]:
        if metric == "n_labels":
            continue
        for imb in args.imbalances:
            plt.plot(
                imbalance_stats[imb]["n_labels"],
                imbalance_stats[imb][metric][args.measure],
                label=f"Imbalance: {imb}",
            )
        plt.title(metric)
        plt.xlabel("Number of labelled samples")
        plt.ylabel(args.measure)
        plt.ylim(0, 0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{data_dir}/metrics_{metric}.png", dpi=300)
        plt.close()
        plt.close()
