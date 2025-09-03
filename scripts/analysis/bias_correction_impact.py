import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arc_tigers.eval.utils import get_metric_stats, get_se_values
from arc_tigers.replay.utils import load_sampling_data
from arc_tigers.sampling.bias import BiasCorrector
from arc_tigers.sampling.metrics import compute_metrics, inverse_softmax, unpack_metrics

METRICS = [
    "accuracy",
    "average_precision",
    "f1_0",
    "f1_1",
    "precision_0",
    "precision_1",
    "recall_0",
    "recall_1",
]


def uncorrect_bias(experiment_dir):
    sample_strat = experiment_dir.split("/")[-1]
    save_dir = f"{experiment_dir}/../bias_uncorrected_{sample_strat}/"
    os.makedirs(save_dir, exist_ok=True)

    sampling_data_files = load_sampling_data(experiment_dir)
    sample_seeds = sampling_data_files["seeds"]
    sampling_runs = sampling_data_files["sample_data"]
    with open(f"{experiment_dir}/stats_full.json") as file:
        sampling_file = json.load(file)

    shutil.copy(f"{experiment_dir}/metrics_full.json", f"{save_dir}/metrics_full.json")

    for sample_data, seed in zip(sampling_runs, sample_seeds, strict=True):
        metrics_df = pd.read_csv(f"{experiment_dir}/metrics_{seed}.csv")
        sample_counts = metrics_df["n"]

        bias_corrector = BiasCorrector(
            N=sample_data["dataset_size"], M=sample_data["n_labels"]
        )
        sample_indices = sample_data["sample_idx"]
        sample_probabilities = sample_data["sample_prob"]

        sample_index = 0
        uncorrected_df = pd.DataFrame()
        for n_samples in sample_counts:
            for p in sample_probabilities[sample_index:n_samples]:
                bias_corrector.compute_weighting_factor(p, sample_index)
                sample_index += 1

            indices = sample_indices[:n_samples]
            # use 1d preds as compute metrics expects logits or binary predictions
            preds = np.array(sampling_file["softmax"])[indices]
            logits = inverse_softmax(preds)
            accuracy = np.array(sampling_file["accuracy"])[indices]
            labels = (np.argmax(logits, axis=-1) == accuracy).astype(int)
            uncorrected_metrics = unpack_metrics(compute_metrics((logits, labels)))
            uncorrected_metrics["n"] = n_samples
            uncorrected_df = pd.concat(
                [uncorrected_df, pd.DataFrame([uncorrected_metrics])], ignore_index=True
            )
        uncorrected_df.to_csv(f"{save_dir}/metrics_{seed}.csv", index=False)

    get_metric_stats(save_dir)


def plot_comparison(corrected, uncorrected, metric, savedir):
    plt.plot(corrected, label="Corrected")
    plt.plot(uncorrected, label="Uncorrected")
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    plt.xlabel("Samples")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"{savedir}/{metric}_comparison.png")
    plt.close()


def compare_bias_correction(experiment_dir):
    uncorrected_dir = (
        f"{experiment_dir}/../bias_uncorrected_{experiment_dir.split('/')[-1]}/"
    )

    corrected_se_vals = get_se_values(experiment_dir)
    uncorrected_se_vals = get_se_values(uncorrected_dir)

    for metric in METRICS:
        corrected_metric_mean = np.sqrt(np.mean(corrected_se_vals[metric], axis=0))
        uncorrected_metric_mean = np.sqrt(np.mean(uncorrected_se_vals[metric], axis=0))

        plot_comparison(
            corrected=(corrected_metric_mean),
            uncorrected=(uncorrected_metric_mean),
            metric=metric,
            savedir=uncorrected_dir,
        )


if __name__ == "__main__":
    example_dir = (
        "outputs/reddit_dataset_12/one-vs-all/football/42_05/distilbert/"
        "default/eval_outputs/05/info_gain_lightgbm"
    )
    # uncorrect_bias(example_dir)
    compare_bias_correction(example_dir)
