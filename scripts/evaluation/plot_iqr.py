import argparse

import matplotlib.pyplot as plt
from eval_sampling import get_metric_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sampling metrics.")
    parser.add_argument(
        "data_dir", type=str, help="Directory containing the metrics files."
    )
    args = parser.parse_args()
    imbalances = [0.01, 0.1, 0.25, 0.5]

    data_dir = args.data_dir
    imbalance_stats = {}
    for imb in imbalances:
        imb_dir = (
            f"{data_dir}/imbalanced_random_sampling_outputs_{str(imb).replace('.', '')}"
        )
        imbalance_stats[imb] = get_metric_stats(imb_dir, plot=False)

    for metric in imbalance_stats[imbalances[0]]:
        if metric == "n_labels":
            continue
        for imb in imbalances:
            plt.plot(
                imbalance_stats[imb]["n_labels"],
                imbalance_stats[imb][metric]["IQR"],
                label=f"Imbalance: {imb}",
            )
        plt.title(metric)
        plt.xlabel("Number of labelled samples")
        plt.ylabel("IQR")
        plt.ylim(0, 0.8)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{data_dir}/metrics_{metric}.png", dpi=300)
        plt.close()
