import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns


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
        plt.ylabel(metric)
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
                label=strategy_name,
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
        plt.axhline(
            y=0,
            color="k",
            linestyle="--",
            alpha=0.5,
        )

        for strategy_name in strategy_names:
            mse = metric_stats[strategy_name][metric]["mse"]
            x_vals = metric_stats[strategy_name]["n_labels"][1:]  # type: ignore[index]
            y_vals = mse[1:]

            plt.plot(x_vals, y_vals, label=strategy_name, linestyle="-", linewidth=2)

        plt.ylabel(f"MSE from full test {metric}")
        plt.xlabel("Number of labelled samples")
        plt.legend()
        plt.title(f"Sampling Strategy Comparison: Sampling Error {metric}")
        plt.tight_layout()
        plt.savefig(f"{save_dir}/mse/{metric}.png", dpi=300)
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


def class_pct_summary_by_size(df: pd.DataFrame) -> pd.DataFrame:
    """Create summary statistics by sampling method and sample size."""
    if df.empty:
        return pd.DataFrame()

    # Group by sampling method and sample size
    summary = (
        df.groupby(["sampling_method", "sample_size"])
        .agg(
            {
                "positive_class_pct": ["mean", "std", "count"],
                "negative_class_pct": ["mean", "std"],
                "n_class_0": "mean",
                "n_class_1": "mean",
            }
        )
        .round(3)
    )

    # Flatten column names
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    return summary.reset_index()


def class_percentage_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean table suitable for LaTeX output like the aggregate analysis
    scripts.
    """
    if df.empty:
        return pd.DataFrame()

    # Create pivot table with sampling methods as rows and sample sizes as columns
    pivot_df = df.pivot(
        index="sampling_method", columns="sample_size", values="positive_class_pct_mean"
    ).round(2)

    # Add a column with method names for cleaner display
    pivot_df.index.name = None  # Remove index name

    return pivot_df


def save_latex_summary_table(df: pd.DataFrame, output_file: str, caption: str = ""):
    """Save a summary table as LaTeX like the other analysis scripts."""
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
