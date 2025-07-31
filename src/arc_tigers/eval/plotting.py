import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


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
    os.makedirs(f"{figures_dir}/comparison", exist_ok=True)

    # Get the base model results
    base_stats = replay_results.get("base")
    if not base_stats:
        print("No base model results found!")
        return

    # Get list of metrics (excluding 'n_labels' and class count metrics)
    metrics = [m for m in base_stats[0] if m != "n_labels" and "n_class" not in m]

    # Plot 1: Raw performance comparison
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Plot base model performance
        base_metric_stats = base_stats[0]
        x_vals = base_metric_stats["n_labels"]
        mean_vals = base_metric_stats[metric]["mean"]
        std_vals = base_metric_stats[metric]["std"]

        plt.plot(
            x_vals,
            mean_vals,
            label="Base Model (Original)",
            linestyle="-",
        )
        plt.fill_between(x_vals, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3)

        # Plot transfer results for each model
        transfer_models = [k for k in replay_results if k != "base"]

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
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(f"Transfer Performance: {metric}")
        plt.tight_layout()
        plt.savefig(f"{figures_dir}/raw/{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()

    # Plot 2: Transfer efficiency (ratio to base model)
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        base_metric_stats = base_stats[0]
        base_x_vals = base_metric_stats["n_labels"]
        base_mean_vals = base_metric_stats[metric]["mean"]

        # Baseline at 0 (same as base model)
        plt.axhline(
            y=0,
            color="black",
            linestyle="-",
            alpha=0.5,
            label="Base",
        )

        # Plot transfer efficiency for each model
        for model_name in transfer_models:
            if replay_results[model_name] is None:
                continue

            transfer_stats = replay_results[model_name][0]
            transfer_x_vals = transfer_stats["n_labels"]
            transfer_mean_vals = transfer_stats[metric]["mean"]

            # Calculate efficiency ratio (transfer performance / base performance)
            # Interpolate base values to match transfer x values if needed
            if not np.array_equal(base_x_vals, transfer_x_vals):
                base_interp = np.interp(transfer_x_vals, base_x_vals, base_mean_vals)
            else:
                base_interp = base_mean_vals

            difference = base_interp - transfer_mean_vals

            plt.plot(
                transfer_x_vals,
                difference,
                label=f"Transfer from {model_name}",
                linestyle="-",
            )

        plt.ylabel("Transfer Efficiency (ratio to base)")
        plt.xlabel("Number of labelled samples")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.title(f"Transfer Efficiency: {metric}")
        plt.tight_layout()
        plt.savefig(
            f"{figures_dir}/comparison/{metric}_efficiency.png",
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
