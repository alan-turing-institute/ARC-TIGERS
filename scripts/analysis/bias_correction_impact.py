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
    "average_precision",
    "f1_0",
    "f1_1",
]

SAMPLERS = ["accuracy_lightgbm", "info_gain_lightgbm", "minority", "ssepy", "isolation"]
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

    # get_metric_stats(experiment_dir)
    # get_metric_stats(save_dir)

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

        corrected_dict[metric].append(corrected_se_vals[metric][:, 2:].flatten())
        uncorrected_dict[metric].append(uncorrected_se_vals[metric][:, 2:].flatten())

    return {"corrected": corrected_dict, "uncorrected": uncorrected_dict}


def run_analysis():
    # Create dictionaries to store results for each sampler and bias correction setting
    all_results = {}

    for imbalance in IMBALANCE_LEVELS:
        for sampler in SAMPLERS:
            uncorrected_dict = {metric: [] for metric in METRICS}
            corrected_dict = {metric: [] for metric in METRICS}
            for model in MODELS:
                example_dir = (
                    f"{base_dir}/{model[0]}/{model[1]}/eval_outputs/"
                    f"{imbalance}/{sampler}"
                )
                # uncorrect_bias(example_dir)
                new_results = compare_bias_correction(
                    example_dir,
                    corrected_dict=corrected_dict,
                    uncorrected_dict=uncorrected_dict,
                )
                uncorrected_dict = new_results["uncorrected"]
                corrected_dict = new_results["corrected"]

            print(f"\nSample Strategy: {sampler}, Imbalance: {imbalance}")

            # Format imbalance level for display
            imbalance_pct = int(float(imbalance[:1] + "." + imbalance[1:]) * 100)

            # Store results for this sampler/imbalance combination
            if sampler not in all_results:
                all_results[sampler] = {}

            all_results[sampler][imbalance] = {"corrected": {}, "uncorrected": {}}

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

                bootstrap_corrected = (
                    boot_corrected.bootstrap_distribution.mean(),
                    (
                        boot_corrected.confidence_interval.low,
                        boot_corrected.confidence_interval.high,
                    ),
                )

                corrected_formatted = (
                    f"${bootstrap_corrected[0]:.4f}"
                    f"^{{{bootstrap_corrected[1][1]:.4f}}}"
                    f"_{{{bootstrap_corrected[1][0]:.4f}}}$"
                )

                bootstrap_uncorrected = (
                    boot_uncorrected.bootstrap_distribution.mean(),
                    (
                        boot_uncorrected.confidence_interval.low,
                        boot_uncorrected.confidence_interval.high,
                    ),
                )

                uncorrected_formatted = (
                    f"${bootstrap_uncorrected[0]:.4f}"
                    f"^{{{bootstrap_uncorrected[1][1]:.4f}}}"
                    f"_{{{bootstrap_uncorrected[1][0]:.4f}}}$"
                )
                all_results[sampler][imbalance]["corrected"][metric] = (
                    corrected_formatted
                )
                all_results[sampler][imbalance]["uncorrected"][metric] = (
                    uncorrected_formatted
                )

    # Create the final DataFrame with imbalance levels as columns
    column_names = ["Sample Strategy", "BC"]
    for imbalance in IMBALANCE_LEVELS:
        imbalance_pct = int(float(imbalance[:1] + "." + imbalance[1:]) * 100)
        for metric in METRICS:
            metric_name = METRIC_NAME_MAP.get(metric, metric)
            column_names.append(f"{metric_name} ({imbalance_pct}\\%)")

    bias_correction_df = pd.DataFrame(columns=column_names)

    # Populate the DataFrame
    for sampler in SAMPLERS:
        # Create rows for corrected and uncorrected
        corrected_row = {
            "Sample Strategy": SAMPLE_STRAT_NAME_MAP.get(sampler, sampler),
            "BC": "Yes",
        }
        uncorrected_row = {
            "Sample Strategy": SAMPLE_STRAT_NAME_MAP.get(sampler, sampler),
            "BC": "No",
        }

        for imbalance in IMBALANCE_LEVELS:
            imbalance_pct = int(float(imbalance[:1] + "." + imbalance[1:]) * 100)
            for metric in METRICS:
                metric_name = METRIC_NAME_MAP.get(metric, metric)
                col_name = f"{metric_name} ({imbalance_pct}\\%)"
                corrected_row[col_name] = all_results[sampler][imbalance]["corrected"][
                    metric
                ]
                uncorrected_row[col_name] = all_results[sampler][imbalance][
                    "uncorrected"
                ][metric]

        bias_correction_df = pd.concat(
            [bias_correction_df, pd.DataFrame([corrected_row, uncorrected_row])],
            ignore_index=True,
        )

    # Save the combined results as a single table
    bias_correction_df.to_csv(
        f"{table_dir}/aggregated_sampling_results_all_imbalances.csv",
        index=False,
    )

    # Calculate number of columns for LaTeX formatting
    num_metric_cols = len(METRICS) * len(IMBALANCE_LEVELS)
    bias_correction_df.to_latex(
        f"{table_dir}/aggregated_sampling_results_all_imbalances.tex",
        index=False,
        column_format="ll" + "c" * num_metric_cols,
        label="tab:bias_correction_sampling_strats_all_imbalances",
        caption=(
            "The impact of bias correction on different sampling strategies across "
            "all imbalance levels. Results are aggregated over 80 runs across "
            "4 models and bootstrapped with 1000 resamples."
        ),
    )


if __name__ == "__main__":
    base_dir = "outputs/reddit_dataset_12/one-vs-all/football/42_05/"
    table_dir = f"{base_dir}/tables/bias_correction"
    os.makedirs(table_dir, exist_ok=True)
    run_analysis()
