import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from cycler import cycler
from scipy.stats import bootstrap

METRIC_GROUPS = {
    "f1_1": ["f1_1"],
    "f1_0": ["f1_0"],
    "average_precision": ["average_precision"],
}

METRIC_GROUP_NAME_MAP = {
    "f1_1": "F1 (Minority)",
    "f1_0": "F1 (Majority)",
    "average_precision": "Average Precision",
}

MODEL_NAME_MAP = {
    "distilbert": "DistilBERT",
    "ModernBERT": "ModernBERT",
    "gpt2": "GPT-2",
    "zero-shot": "Zero-Shot",
}

METRIC_NAME_MAP = {
    "accuracy": "Accuracy",
    "loss": "Loss",
    "f1_0": "F1 (Majority)",
    "f1_1": "F1 (Minority)",
    "precision_0": "Precision (Majority)",
    "precision_1": "Precision (Minority)",
    "recall_0": "Recall (Majority)",
    "recall_1": "Recall (Minority)",
    "average_precision": "Average Precision",
}

SHORTENED_METRIC_NAME_MAP = {
    "accuracy": "Acc.",
    "loss": "Loss",
    "f1_0": "F1 (Maj.)",
    "f1_1": "F1 (Min.)",
    "precision_0": "Prec. (Maj.)",
    "precision_1": "Prec. (Min.)",
    "recall_0": "Rec. (Maj.)",
    "recall_1": "Rec. (Min.)",
    "average_precision": "Avg. Prec.",
}

SAMPLE_STRAT_NAME_MAP = {
    "random": "Random Uniform",
    "ssepy": "ssepy",
    "minority": "Minority",
    "isolation": "Isolation",
    "info_gain_lightgbm": "Cross-Entropy",
    "accuracy_lightgbm": "Accuracy",
}


def plot_replay_results(replay_results, save_dir):
    """
    Plot transfer results showing how different models perform when using
    sampling sequences from the base model.
    """
    if not replay_results:
        print("No replay results to plot!")
        return

    # Create save directory for figures
    figures_dir = f"{save_dir}/transfer_analysis"
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(f"{figures_dir}/raw", exist_ok=True)
    os.makedirs(f"{figures_dir}/mse", exist_ok=True)
    os.makedirs(f"{figures_dir}/difference", exist_ok=True)

    # Get the base model results
    base_stats = replay_results.get("base")
    if not base_stats:
        print("No base model results found!")
        return

    # Get list of metrics (excluding 'n_labels' and class count metrics)
    metrics = [m for m in base_stats[0] if m != "n_labels" and "n_class" not in m]
    full_metrics = base_stats[1]

    transfer_models = [
        k for k in replay_results if k not in ["base", "random_baseline"]
    ]

    # Plot 1: Raw performance comparison
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        # Plot base model performance
        base_metric_stats = base_stats[0]
        x_vals = base_metric_stats["n_labels"]
        mean_vals = base_metric_stats[metric]["mean"]
        std_vals = base_metric_stats[metric]["std"]

        plt.axhline(
            y=full_metrics[metric],
            color="black",
            linestyle="--",
            alpha=0.95,
            label="Full Test Set",
        )

        # Plot random baseline if available
        if (
            "random_baseline" in replay_results
            and replay_results["random_baseline"] is not None
        ):
            random_stats = replay_results["random_baseline"][0]
            x_vals_random = random_stats["n_labels"][1:]
            mean_vals_random = random_stats[metric]["mean"][1:]
            std_vals_random = random_stats[metric]["std"][1:]

            plt.plot(
                x_vals_random,
                mean_vals_random,
                label="Random baseline",
                linestyle=":",
                color="gray",
            )
            plt.fill_between(
                x_vals_random,
                mean_vals_random - std_vals_random,
                mean_vals_random + std_vals_random,
                alpha=0.2,
            )

        plt.plot(
            x_vals,
            mean_vals,
            label="Original model",
            linestyle="-",
        )
        plt.fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)

        # Plot transfer results for each model

        for model_name in transfer_models:
            if replay_results[model_name] is None:
                continue

            transfer_stats = replay_results[model_name][0]
            x_vals_transfer = transfer_stats["n_labels"]
            mean_vals_transfer = transfer_stats[metric]["mean"]
            std_vals_transfer = transfer_stats[metric]["std"]

            plt.plot(
                x_vals_transfer,
                mean_vals_transfer,
                label=f"Transfer to {model_name}",
                linestyle="-",
            )
            plt.fill_between(
                x_vals_transfer,
                mean_vals_transfer - std_vals_transfer,
                mean_vals_transfer + std_vals_transfer,
                alpha=0.2,
            )

        plt.ylabel(f"{metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend(loc="upper right")
        plt.title(f"Transfer Performance: {metric}")
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/raw/{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot 2: Transfer efficiency (difference to full test set)
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        # Get full test set performance as baseline
        full_test_value = full_metrics[metric]

        # Baseline at 0 (same as full test set)
        # plt.axhline(
        #     y=0,
        #     color="black",
        #     linestyle="--",
        #     alpha=0.9,
        #     label="Full Test Set",
        # )
        # plot random baseline if available
        if (
            "random_baseline" in replay_results
            and replay_results["random_baseline"] is not None
        ):
            random_stats = replay_results["random_baseline"][0]
            x_vals_random = random_stats["n_labels"][1:]
            mean_vals_random = random_stats[metric]["mean"][1:]

            # Calculate difference from full test set performance for random baseline
            random_difference = mean_vals_random - full_test_value

            plt.plot(
                x_vals_random,
                random_difference,
                label="Random baseline",
                linestyle=":",
                linewidth=2,
                color="gray",
            )

        # Plot base model efficiency
        base_metric_stats = base_stats[0]
        base_x_vals = base_metric_stats["n_labels"][1:]
        base_mean_vals = base_metric_stats[metric]["mean"][1:]

        # Calculate difference from full test set performance for base model
        base_difference = base_mean_vals - full_test_value

        plt.plot(
            base_x_vals,
            base_difference,
            label="Original model",
            linestyle="-",
            linewidth=2,
        )

        # Plot transfer efficiency for each model
        for model_name in transfer_models:
            if replay_results[model_name] is None:
                continue

            transfer_stats = replay_results[model_name][0]
            transfer_x_vals = transfer_stats["n_labels"]
            transfer_mean_vals = transfer_stats[metric]["mean"]

            # Calculate difference from full test set performance
            difference = transfer_mean_vals - full_test_value

            plt.plot(
                transfer_x_vals,
                difference,
                label=f"Transfer from {model_name}",
                linestyle="-",
            )

        plt.ylabel(f"Difference from full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend(loc="upper right")
        plt.title(f"Transfer Efficiency: {metric}")
        plt.tight_layout()
        plt.savefig(
            f"{figures_dir}/difference/{metric}_efficiency.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # plot MSE for each model
    for metric in metrics:
        plt.figure(figsize=(8, 6))

        # plt.axhline(
        #     y=0,
        #     color="black",
        #     linestyle="--",
        #     alpha=0.9,
        #     label="Perfect Match to Full Test",
        # )
        # plot random baseline if available
        if (
            "random_baseline" in replay_results
            and replay_results["random_baseline"] is not None
        ):
            random_stats = replay_results["random_baseline"][0]
            x_vals_random = random_stats["n_labels"][1:]

            if "mse" in random_stats[metric]:
                random_mse = random_stats[metric]["mse"][1:]
            else:
                # Calculate MSE manually if not available
                mean_vals_random = random_stats[metric]["mean"]
                random_mse = ((mean_vals_random - full_test_value) ** 2)[1:]

            plt.plot(
                x_vals_random,
                random_mse,
                label="Random baseline",
                linestyle=":",
                linewidth=2,
                color="gray",
            )

        # Plot base model MSE
        base_metric_stats = base_stats[0]
        base_x_vals = base_metric_stats["n_labels"][1:]

        if "mse" in base_metric_stats[metric]:
            base_mse = base_metric_stats[metric]["mse"][1:]
            plt.plot(
                base_x_vals,
                base_mse,
                label="Original model",
                linestyle="-",
                linewidth=2,
            )
        else:
            # Calculate MSE manually if not available
            base_mean_vals = base_metric_stats[metric]["mean"][1:]
            base_mse = (base_mean_vals - full_test_value) ** 2
            plt.plot(
                base_x_vals,
                base_mse,
                label="Original model",
                linestyle="-",
                linewidth=2,
            )

        # Plot transfer model MSE
        for model_name in transfer_models:
            if replay_results[model_name] is None:
                continue

            transfer_stats = replay_results[model_name][0]
            transfer_x_vals = transfer_stats["n_labels"]

            if "mse" in transfer_stats[metric]:
                transfer_mse = transfer_stats[metric]["mse"]
            else:
                # Calculate MSE manually if not available
                transfer_mean_vals = transfer_stats[metric]["mean"]
                transfer_mse = (transfer_mean_vals - full_test_value) ** 2

            plt.plot(
                transfer_x_vals,
                transfer_mse,
                label=f"Transfer from {model_name}",
                linestyle="-",
            )

        plt.ylabel(f"MSE from full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend(loc="upper right")
        plt.title(f"Transfer Sampling Error: {metric}")
        plt.tight_layout()
        plt.savefig(
            f"{figures_dir}/mse/{metric}_mse.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    print(f"Transfer analysis plots saved to {save_dir}")


def plot_difference(
    save_dir: str,
    stats: dict[
        str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
    ],
    full_metrics: dict[str, int | float],
):
    for metric in stats:
        if metric == "n_labels":
            continue

        if "n_class" in metric:
            continue

        plt.axhline(y=0, color="k", linestyle="--", alpha=0.2)
        diff = stats[metric]["mean"] - full_metrics[metric]
        diff_min = (stats[metric]["mean"] - stats[metric]["std"]) - full_metrics[metric]
        diff_max = (stats[metric]["mean"] + stats[metric]["std"]) - full_metrics[metric]

        plt.plot(
            stats["n_labels"],
            diff,
            label="Mean (labelled subset)",
            linestyle="-",
            linewidth=2,
        )
        plt.fill_between(
            stats["n_labels"],
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


def plot_metrics(
    save_dir: str,
    stats: dict[
        str, npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]]
    ],
    full_metrics: dict[str, int | float],
    quantiles: list[float] | None = None,
    quantile_colors: list[str] | None = None,
):
    """Plot how metric stats vary with the number of labelled samples.

    Args:
        save_dir: Directory to save the plots.
        stats: Dictionary containing the metric stats.
        full_metrics: Dictionary containing the metrics of the full dataset.
        quantiles: List of quantiles to plot. If None, defaults to [0.025, 0.25].
        quantile_colors: List of colors for the quantiles. If None, defaults to ["r",
            "orange"].
    """
    if quantile_colors is None:
        quantile_colors = ["r", "orange"]
    if quantiles is None:
        quantiles = [0.025, 0.25]

    limits_dict = {
        "accuracy": (0.5, 1.0),
        "loss": (0.0, 2.0),
        "n_class_0": (0, np.max(stats["n_labels"])),
        "n_class_1": (0, np.max(stats["n_labels"])),
    }

    for metric in stats:
        if metric == "n_labels":
            continue
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
            stats["n_labels"],
            stats[metric]["mean"],
            label="Mean (labelled subset)",
            color="blue",
            linestyle="-",
            linewidth=2,
        )
        for quantile, color in zip(quantiles, quantile_colors, strict=True):
            plt.fill_between(
                stats["n_labels"],
                stats[metric][f"quantile_{quantile * 100}"],
                stats[metric][f"quantile_{(1 - quantile) * 100}"],
                color=color,
                alpha=0.2,
                label=f"Quantiles ({quantile * 100} - {(1 - quantile) * 100})",
            )
        y_min = limits_dict.get(metric, (-0.1, 1.1))[0]
        y_max = limits_dict.get(metric, (-0.1, 1.1))[1]
        plt.ylim(y_min, y_max)
        plt.ylabel(METRIC_NAME_MAP.get(metric, metric))
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/metrics_{metric}.png", dpi=300)
        plt.close()


def model_comparison_mse(
    save_dir: str,
    metric_stats: dict[
        str,
        dict[
            str,
            npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]],
        ],
    ],
):
    model_names = list(metric_stats.keys())
    metrics = list(metric_stats[model_names[0]].keys())
    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue
        plt.figure(figsize=(7, 5))
        plt.axhline(
            y=0,
            color="k",
            linestyle="--",
            alpha=0.5,
        )
        for model_name in model_names:
            mse = metric_stats[model_name][metric]["mse"]
            mse = metric_stats[model_name][metric]["mse"]
            # ignore first index as mse is very high at low label counts
            x_vals = metric_stats[model_name]["n_labels"][1:]  # type: ignore[index]
            y_vals = mse[1:]
            plt.plot(x_vals, y_vals, label=model_name, linestyle="-", linewidth=2)

        plt.ylabel(f"MSE from full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Model Comparison: {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mse/{metric}.png", dpi=300)
        plt.close()


def model_comparison_raw(
    save_dir: str,
    metric_stats: dict[
        str,
        dict[
            str,
            npt.NDArray[np.integer] | dict[str, npt.NDArray[np.integer | np.floating]],
        ],
    ],
    full_metrics: dict[str, dict[str, int | float]] | None = None,
):
    """Plot the raw metric values for each model."""
    model_names = list(metric_stats.keys())
    metrics = metric_stats[model_names[0]].keys()
    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue
        plt.figure(figsize=(7, 5))

        for model_name in model_names:
            mean_values = metric_stats[model_name][metric]["mean"]
            std_values = metric_stats[model_name][metric]["std"]
            x_vals = metric_stats[model_name]["n_labels"]

            plt.plot(
                x_vals,
                mean_values,
                label=model_name,
                linestyle="-",
                linewidth=2,
                markersize=4,
            )
            plt.fill_between(
                x_vals, mean_values - std_values, mean_values + std_values, alpha=0.2
            )

            # Plot baseline (full test set) for this specific model if available
            if full_metrics and model_name in full_metrics and "n_class" not in metric:
                baseline_value = full_metrics[model_name][metric]
                # Use same color as model line
                model_color = plt.gca().lines[-1].get_color()
                plt.axhline(
                    y=baseline_value,
                    color=model_color,
                    linestyle="--",
                    alpha=0.7,
                    linewidth=1,
                )

        plt.ylabel(f"{metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Model Comparison: {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/raw/{metric}.png", dpi=300)
        plt.close()


def sampling_comparison_raw(
    save_dir: str,
    metric_stats: dict[str, dict[str, npt.NDArray | dict[str, npt.NDArray]]],
    full_metrics: dict[str, dict[str, int | float]] | None = None,
):
    """Plot raw metric values for sampling strategy comparison."""
    strategy_names = list(metric_stats.keys())
    metrics = metric_stats[strategy_names[0]].keys()

    for metric in metrics:
        if metric == "n_labels":
            continue

        plt.figure(figsize=(7, 5))

        # Plot baseline (full test set) if available
        if full_metrics and "n_class" not in metric:
            # Use the baseline from the first strategy (they should all be the same)
            baseline_value = next(iter(full_metrics.values()))[metric]
            plt.axhline(
                y=baseline_value,
                color="k",
                linestyle="--",
                alpha=0.7,
                linewidth=2,
                label="Full Test Set",
            )

        for strategy_name in strategy_names:
            mean_values = metric_stats[strategy_name][metric]["mean"]
            std_values = metric_stats[strategy_name][metric]["std"]
            x_vals = metric_stats[strategy_name]["n_labels"]

            plt.plot(
                x_vals,
                mean_values,
                label=SAMPLE_STRAT_NAME_MAP.get(strategy_name, strategy_name),
                linestyle="-",
                linewidth=2,
                markersize=4,
            )
            plt.fill_between(
                x_vals, mean_values - std_values, mean_values + std_values, alpha=0.2
            )

        plt.ylabel(f"{metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Sampling Strategy Comparison: Raw {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/raw/{metric}.png", dpi=300)
        plt.close()


def sampling_comparison_mse(
    save_dir: str,
    metric_stats: dict[str, dict[str, npt.NDArray | dict[str, npt.NDArray]]],
    imbalance: str,
):
    """Plot MSE (sampling error) for sampling strategy comparison."""
    strategy_names = list(metric_stats.keys())
    metrics = metric_stats[strategy_names[0]].keys()

    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue

        plt.figure(figsize=(7, 5))
        c = plt.get_cmap("Set2").colors
        plt.rcParams["axes.prop_cycle"] = cycler(color=c)
        # plt.axhline(
        #     y=0,
        #     color="k",
        #     linestyle="--",
        #     alpha=0.5,
        # )
        index = 0
        colours = [1, 0, 3, 2, 4]
        actual_colours = [
            (0.3860518578202345, 0.347233231410498, 0.9587675565854633),
            (0.3445203292825778, 0.9058528933329075, 0.3542249551359389),
            (0.9906165268127168, 0.5364405488158809, 0.35015868672286615),
            (0.40669494593387073, 0.8441399379845326, 0.964301804990655),
            (0.9853462883412639, 0.3674190826435139, 0.9724504814304838),
        ]
        for strategy_name in strategy_names:
            linestyle = "--" if strategy_name == "random" else "-"
            linesize = 2.5 if strategy_name == "random" else 2
            linesize = 3
            colour = (
                "black" if strategy_name == "random" else actual_colours[colours[index]]
            )

            mse = metric_stats[strategy_name][metric]["mse"]
            x_vals = metric_stats[strategy_name]["n_labels"][1:]  # type: ignore[index]
            y_vals = np.sqrt(mse[1:])

            plt.plot(
                x_vals,
                y_vals,
                label=SAMPLE_STRAT_NAME_MAP.get(strategy_name, strategy_name),
                linestyle=linestyle,
                linewidth=linesize,
                color=colour,
            )
            if strategy_name != "random":
                index += 1

        plt.ylim(0, plt.ylim()[1])
        plt.ylabel(
            f"RMSE from full test {METRIC_NAME_MAP.get(metric, metric)}",
            fontsize=16,
        )
        plt.xlabel("Number of labelled samples", fontsize=16)
        plt.legend(fontsize=14)
        # plt.title(
        #     "Sampling Strategy Comparison: Sampling Error "
        #     f"{METRIC_NAME_MAP.get(metric, metric)}",
        #     fontsize=15,
        # )
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mse/{metric}_{imbalance}.png", dpi=300)
        plt.close()


def sampling_comparison_improvement(
    save_dir: str,
    metric_stats: dict[str, dict[str, npt.NDArray | dict[str, npt.NDArray]]],
):
    """Plot improvement of sampling strategies relative to random baseline."""
    strategy_names = list(metric_stats.keys())
    metrics = metric_stats[strategy_names[0]].keys()

    # Check if random is available as baseline
    if "random" not in strategy_names:
        print("Warning: 'random' strategy not found, cannot compute improvements")
        return

    for metric in metrics:
        if metric == "n_labels":
            continue
        if "n_class" in metric:
            continue

        plt.figure(figsize=(12, 6))
        plt.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Random")

        # Use random as baseline for comparison
        baseline_mse = metric_stats["random"][metric]["mse"][1:]
        x_vals = metric_stats["random"]["n_labels"][1:]  # type: ignore[index]

        for strategy_name in strategy_names:
            mse = metric_stats[strategy_name][metric]["mse"]
            if strategy_name == "random":
                continue
            # Calculate improvement: baseline_mse / current_mse
            # Values > 1 mean improvement, < 1 mean degradation
            improvement = baseline_mse / mse[1:]
            plt.plot(
                x_vals,
                improvement,
                label=strategy_name,
                linestyle="-",
                linewidth=2,
            )

        plt.ylabel(f"Improvement over random: {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Sampling Strategy Comparison: Improvement in {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/improvement/{metric}.png", dpi=300)
        plt.close()


def confidence_interval_lower(x, confidence_level: float = 0.95) -> float:
    mean = x.mean()
    std_err = x.std() / np.sqrt(len(x))
    t_value = stats.t.ppf((1 + confidence_level) / 2, len(x) - 1)
    return mean - t_value * std_err


def confidence_interval_upper(x, confidence_level: float = 0.95) -> float:
    mean = x.mean()
    std_err = x.std() / np.sqrt(len(x))
    t_value = stats.t.ppf((1 + confidence_level) / 2, len(x) - 1)
    return mean + t_value * std_err


def bootstrap_confidence(x):
    """Compute bootstrap confidence intervals for the given data."""
    bootstrapped = bootstrap((x,), np.mean, confidence_level=0.95, n_resamples=10000)
    return (bootstrapped.confidence_interval.low, bootstrapped.confidence_interval.high)


def class_pct_summary_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics by sampling method and sample size."""
    if df.empty:
        return pd.DataFrame()

    # Group by sampling method and sample size
    summary = (
        df.groupby(["sampling_method", "sample_size"])
        .agg(
            {
                "positive_class_pct": ["mean", "std", "count", bootstrap_confidence],
                "negative_class_pct": ["mean", "std", bootstrap_confidence],
                "n_class_0": "mean",
                "n_class_1": "mean",
            }
        )
        .round(3)
    )

    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    return summary.reset_index()


def generate_grouped_tables(
    grouped_bootstrap_results: dict,
    evaluate_steps: list[int],
    imbalances: list[str],
    sampling_methods: list[str],
    save_dir: str,
    verbose: bool = False,
    groups=METRIC_GROUPS,
) -> None:
    """Generate tables for grouped bootstrap results, one per imbalance level."""

    for imbalance in imbalances:
        if verbose:
            print(f"Generating grouped table for imbalance {imbalance}")

        # Create table for this imbalance
        rows = []

        for sampling_method in sampling_methods:
            strategy_name = SAMPLE_STRAT_NAME_MAP.get(sampling_method, sampling_method)
            row = {"Strategy": strategy_name}

            # Add columns for each metric group and sample count
            for group_name in groups:
                for step_idx, step in enumerate(evaluate_steps):
                    col_name = f"{group_name} - {step}"

                    # Get bootstrap result for this configuration
                    group_key = group_name.lower()
                    results = grouped_bootstrap_results[imbalance][group_key][
                        sampling_method
                    ]
                    result = results[step_idx]
                    row[col_name] = (
                        f"${result[0]:.3f}^"
                        f"{{{result[1][1]:.3f}}}_"
                        f"{{{result[1][0]:.3f}}}$"
                    )  # mean^high_low

            rows.append(row)

        # Create DataFrame and save
        df = pd.DataFrame(rows)

        # Create directory for this imbalance
        imbalance_dir = f"{save_dir}/{imbalance}"
        os.makedirs(imbalance_dir, exist_ok=True)

        # Save as LaTeX and CSV
        latex_file = f"{imbalance_dir}/grouped_bootstrap_table_{imbalance}.tex"
        csv_file = f"{imbalance_dir}/grouped_bootstrap_table_{imbalance}.csv"

        df.to_latex(
            latex_file,
            index=False,
            float_format="%.4f",
            caption=f"Grouped Bootstrap RMSE Results for Imbalance {imbalance}",
            label=f"tab:grouped_bootstrap_{imbalance}",
        )
        df.to_csv(csv_file, index=False)

        if verbose:
            print(f"Saved grouped table for imbalance {imbalance}")
            print(f"LaTeX: {latex_file}")
            print(f"CSV: {csv_file}")
            print(df)
            print()


def class_percentage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean table suitable for LaTeX output like the aggregate analysis
    scripts.
    """
    if df.empty:
        return pd.DataFrame()

    # First, let's debug what columns we have
    print("Available columns in df:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())

    # Create pivot table with sampling methods as rows and sample sizes as columns
    try:
        pivot_df = df.pivot(
            index="sampling_method",
            columns="sample_size",
            values=[
                "positive_class_pct_mean",
                "positive_class_pct_bootstrap_confidence",
            ],
        )

        print("Pivot table shape:", pivot_df.shape)
        print("Pivot table columns:", pivot_df.columns.tolist())

    except Exception as e:
        print(f"Error creating pivot table: {e}")

    # Create a new DataFrame with combined mean and CI strings
    combined_data = {}

    # Get unique sample sizes
    sample_sizes = df["sample_size"].unique()

    for sample_size in sample_sizes:
        mean_col = ("positive_class_pct_mean", sample_size)
        ci = ("positive_class_pct_bootstrap_confidence", sample_size)

        if mean_col in pivot_df.columns and ci in pivot_df.columns:
            # Format as LaTeX with confidence intervals
            combined_data[sample_size] = (
                pivot_df[mean_col].apply(lambda x: f"${x:.2f}")
                + "^{"
                + pivot_df[ci].apply(lambda x: f"{x[1]:.2f}")
                + "}_"
                + "{"
                + pivot_df[ci].apply(lambda x: f"{x[0]:.2f}")
                + "}$"
            )
        else:
            print(f"Missing columns for sample size {sample_size}")
            # Fallback to just mean values if available
            if mean_col in pivot_df.columns:
                combined_data[sample_size] = pivot_df[mean_col].apply(
                    lambda x: f"{x:.2f}"
                )
            else:
                combined_data[sample_size] = "N/A"

    # Create final DataFrame
    result_df = pd.DataFrame(combined_data, index=pivot_df.index)

    # Clean up the index
    result_df.index.name = None

    # Map sampling method names to more readable versions
    result_df.index = result_df.index.map(lambda x: SAMPLE_STRAT_NAME_MAP.get(x, x))

    print("Final table shape:", result_df.shape)
    print("Final table:")
    print(result_df.head())

    return result_df


def save_latex_summary_table(df: pd.DataFrame, output_file: str, caption: str = ""):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        latex_str = df.to_latex(
            float_format="%.2f",
            caption=caption,
            label=f"tab:{output_path.stem}",
        )
        f.write(latex_str)


def plot_class_distribution_by_size(df: pd.DataFrame, output_dir: str, imbalance: str):
    """Create separate plots showing class distribution vs sample size."""

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Get summary data
    summary_df = class_pct_summary_by_size(df)

    if summary_df.empty:
        print("No data available for plotting")
        return None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Plot 1: Positive class percentage vs sample size
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for method in summary_df["sampling_method"].unique():
        method_data = summary_df[summary_df["sampling_method"] == method]

        ax1.plot(
            method_data["sample_size"],
            method_data["positive_class_pct_mean"],
            marker="o",
            label=method,
            linewidth=2,
            markersize=4,
        )

        # Add error bars (shaded region)
        if "positive_class_pct_std" in method_data.columns:
            ax1.fill_between(
                method_data["sample_size"],
                method_data["positive_class_pct_mean"]
                - method_data["positive_class_pct_std"],
                method_data["positive_class_pct_mean"]
                + method_data["positive_class_pct_std"],
                alpha=0.2,
            )

    ax1.set_xlabel("Sample Size", fontsize=12)
    ax1.set_ylabel("Positive Class Percentage (%)", fontsize=12)
    ax1.set_title(
        f"Positive Class % vs Sample Size (Imbalance {imbalance})", fontsize=14
    )
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    plt.tight_layout()

    # Save Plot 1
    plot1_file = (
        output_path / f"positive_class_pct_vs_sample_size_imbalance_{imbalance}.png"
    )
    plt.savefig(plot1_file, dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Standard deviation vs sample size (sampling variability)
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for method in summary_df["sampling_method"].unique():
        method_data = summary_df[summary_df["sampling_method"] == method]

        if "positive_class_pct_std" in method_data.columns:
            ax2.plot(
                method_data["sample_size"],
                method_data["positive_class_pct_std"],
                marker="s",
                label=method,
                linewidth=2,
                markersize=4,
            )

    ax2.set_xlabel("Sample Size", fontsize=12)
    ax2.set_ylabel("Standard Deviation (%)", fontsize=12)
    ax2.set_title(
        f"Sampling Variability vs Sample Size (Imbalance {imbalance})", fontsize=14
    )
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()

    # Save Plot 2
    plot2_file = (
        output_path / f"sampling_variability_vs_sample_size_imbalance_{imbalance}.png"
    )
    plt.savefig(plot2_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figures saved to: {output_path}/figures/")
    print(f"Tables saved to: {output_path}/tables/")

    return fig1, fig2


def generate_all_metrics_rmse_tables(
    all_metric_stats: dict[
        str, dict[str, dict[str, tuple[float, tuple[float, float]]]]
    ],
    metrics: list[str],
    tables_dir: str,
):
    os.makedirs(tables_dir, exist_ok=True)
    metric_names = [
        f"\\textbf{{{SHORTENED_METRIC_NAME_MAP.get(metric, metric)}}}"
        for metric in metrics
    ]
    """Generate a table with RMSE for all metrics."""
    for imbalance, imbalance_dict in all_metric_stats.items():
        df = pd.DataFrame(columns=["\\textbf{{Sampling Strategy}}", *metric_names])
        for sampling_strategy in imbalance_dict:
            row = [SAMPLE_STRAT_NAME_MAP.get(sampling_strategy, sampling_strategy)]
            if metrics != list(imbalance_dict[sampling_strategy].keys()):
                err_msg = (
                    "Metrics in imbalance_dict do not match provided metrics list."
                )
                raise ValueError(err_msg)
            for _, values in imbalance_dict[sampling_strategy].items():
                mean_val, ci = values
                row.append(f"${mean_val:.3f}_{{{ci[0]:.3f}}}^{{{ci[1]:.3f}}}$")
            df.loc[len(df)] = row
        df.to_latex(
            f"{tables_dir}/all_metric_rmse_table_{imbalance}.tex",
            index=False,
            float_format="%.3f",
            column_format="l" + "c" * len(metrics),
            caption=(
                "RMSE for all metrics at imbalance "
                f"{int(float(imbalance[:1] + '.' + imbalance[1:]) * 100)}\\%. Metrics "
                "are aggregated across all models and results are bootstrapped to a "
                "confidence interval of 95\\%"
            ),
            label=f"tab:all_metric_rmse_{imbalance}",
            position="h!",
        )
        # Read the generated LaTeX file and modify it to add resizebox
        with open(f"{tables_dir}/all_metric_rmse_table_{imbalance}.tex") as f:
            content = f.read()

        # Add resizebox wrapper around the tabular environment
        content = content.replace(
            "\\begin{tabular}", "\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}"
        )
        content = content.replace("\\end{tabular}", "\\end{tabular}\n}")

        # Write the modified content back to the file
        with open(f"{tables_dir}/all_metric_rmse_table_{imbalance}.tex", "w") as f:
            f.write(content)
        df.to_csv(f"{tables_dir}/all_metric_rmse_table_{imbalance}.csv", index=False)
