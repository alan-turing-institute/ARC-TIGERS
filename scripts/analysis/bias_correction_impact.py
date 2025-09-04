import json
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from arc_tigers.eval.aggregate_utils import sqrt_mean
from arc_tigers.eval.plotting import METRIC_NAME_MAP, SAMPLE_STRAT_NAME_MAP
from arc_tigers.eval.utils import get_se_values
from arc_tigers.replay.utils import load_sampling_data
from arc_tigers.sampling.metrics import compute_metrics, inverse_softmax, unpack_metrics

METRICS = [
    "accuracy",
    "loss",
    "average_precision",
    "f1_0",
    "f1_1",
]

SAMPLERS = ["accuracy_lightgbm", "info_gain_lightgbm", "minority", "ssepy"]
MODELS = [
    ("distilbert", "default"),
    ("gpt2", "default"),
    ("ModernBERT", "short"),
    ("zero-shot", "default"),
]
IMBALANCE_LEVELS = ["05", "01", "001"]


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
        sample_indices = sample_data["sample_idx"]
        uncorrected_df = pd.DataFrame()
        for n_samples in sample_counts:
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

    return sample_counts


def plot_comparison(corrected, uncorrected, metric, savedir):
    plt.plot(corrected, label="Corrected")
    plt.plot(uncorrected, label="Uncorrected")
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.4)
    plt.xlabel("Samples")
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(f"{savedir}/{metric}_comparison.png")
    plt.close()


def compare_bias_correction(experiment_dir, corrected_dict, uncorrected_dict):
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

        corrected_dict[metric].append(corrected_se_vals[metric].flatten())
        uncorrected_dict[metric].append(uncorrected_se_vals[metric].flatten())

    return {"corrected": corrected_dict, "uncorrected": uncorrected_dict}


if __name__ == "__main__":
    base_dir = "outputs/reddit_dataset_12/one-vs-all/football/42_05/"
    table_dir = f"{base_dir}/tables/bias_correction"
    os.makedirs(table_dir, exist_ok=True)

    for imbalance in IMBALANCE_LEVELS:
        bias_correction_df = pd.DataFrame(
            columns=[
                "Sample Strategy",
                "BC",
                *[METRIC_NAME_MAP.get(metric, metric) for metric in METRICS],
            ]
        )
        for sampler in SAMPLERS:
            uncorrected_dict = {metric: [] for metric in METRICS}
            corrected_dict = {metric: [] for metric in METRICS}
            for model in MODELS:
                example_dir = (
                    f"{base_dir}/{model[0]}/{model[1]}/eval_outputs/"
                    f"{imbalance}/{sampler}"
                )
                uncorrect_bias(example_dir)
                new_results = compare_bias_correction(
                    example_dir,
                    corrected_dict=corrected_dict,
                    uncorrected_dict=uncorrected_dict,
                )
                uncorrected_dict = new_results["uncorrected"]
                corrected_dict = new_results["corrected"]

            bootstrap_corrected = {}
            bootstrap_uncorrected = {}
            print(f"\nSample Strategy: {sampler}, Imbalance: {imbalance}")
            corrected_df_row = {
                "Sample Strategy": SAMPLE_STRAT_NAME_MAP.get(sampler, sampler),
                "BC": "Yes",
            }
            uncorrected_df_row = {
                "Sample Strategy": SAMPLE_STRAT_NAME_MAP.get(sampler, sampler),
                "BC": "No",
            }
            for metric in METRICS:
                corrected_dict[metric] = np.hstack(corrected_dict[metric]).flatten()
                uncorrected_dict[metric] = np.hstack(uncorrected_dict[metric]).flatten()
                boot_uncorrected = bootstrap(
                    (uncorrected_dict[metric],),
                    sqrt_mean,
                    n_resamples=1000,
                )
                boot_corrected = bootstrap(
                    (corrected_dict[metric],),
                    sqrt_mean,
                    n_resamples=1000,
                )

                bootstrap_corrected[metric] = (
                    boot_corrected.bootstrap_distribution.mean(),
                    (
                        boot_corrected.confidence_interval.low,
                        boot_corrected.confidence_interval.high,
                    ),
                )

                corrected_df_row[METRIC_NAME_MAP.get(metric, metric)] = (
                    f"${bootstrap_corrected[metric][0]:.4f}"
                    f"^{{{bootstrap_corrected[metric][1][1]:.4f}}}"
                    f"_{{{bootstrap_corrected[metric][1][0]:.4f}}}$"
                )

                bootstrap_uncorrected[metric] = (
                    boot_uncorrected.bootstrap_distribution.mean(),
                    (
                        boot_uncorrected.confidence_interval.low,
                        boot_uncorrected.confidence_interval.high,
                    ),
                )

                uncorrected_df_row[METRIC_NAME_MAP.get(metric, metric)] = (
                    f"${bootstrap_uncorrected[metric][0]:.4f}"
                    f"^{{{bootstrap_uncorrected[metric][1][1]:.4f}}}"
                    f"_{{{bootstrap_uncorrected[metric][1][0]:.4f}}}$"
                )
            bias_correction_df = pd.concat(
                [
                    bias_correction_df,
                    pd.DataFrame([corrected_df_row, uncorrected_df_row]),
                ],
                ignore_index=True,
            )
        bias_correction_df.to_csv(
            f"{table_dir}/aggregated_sampling_results_{imbalance}.csv",
            index=False,
        )
        bias_correction_df.to_latex(
            f"{table_dir}/aggregated_sampling_results_{imbalance}.tex",
            index=False,
            column_format="ll" + "c" * len(METRICS),
            label=f"tab:bias_correction_sampling_strats_{imbalance}",
            caption=(
                "The impact of bias correction on different sampling strategies. "
                f"Results are taken at an imbalance of "
                f"{int(float(imbalance[:1] + '.' + imbalance[1:]) * 100)}\\% and are "
                "aggregated over 80 runs across 4 models and "
                "bootstrapped with 1000 resamples."
            ),
        )
