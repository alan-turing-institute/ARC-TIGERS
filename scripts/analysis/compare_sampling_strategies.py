import argparse
import os
from glob import glob

from arc_tigers.eval.plotting import (
    sampling_comparison_improvement,
    sampling_comparison_mse,
    sampling_comparison_raw,
)
from arc_tigers.eval.utils import get_metric_stats


def find_sampling_strategy_directories(
    base_path: str, model: str, test_imbalance: str, sampling_strategies: list[str]
) -> dict[str, str]:
    """
    Find the sampling strategy directories for a given model.

    Args:
        base_path: Base path like "outputs/reddit_dataset_12/one-vs-all/football/42_05"
        model: Model name to search for
        test_imbalance: Test imbalance level (e.g., "05", "01", "001")
        sampling_strategies: List of sampling strategy names
            (e.g., ["random", "distance"])

    Returns:
        Dictionary mapping strategy names to their directory paths
    """
    strategy_paths = {}

    for strategy in sampling_strategies:
        # Look for pattern: base_path/model/*/eval_outputs/test_imbalance/strategy
        pattern = f"{base_path}/{model}/*/eval_outputs/{test_imbalance}/{strategy}"
        matches = glob(pattern)

        if matches:
            # Take the first match - this is the full path to the sampling
            # strategy directory
            strategy_path = matches[0]
            strategy_paths[strategy] = strategy_path
        else:
            print(
                f"Warning: No eval_outputs found for model '{model}' "
                f"with test imbalance '{test_imbalance}' "
                f"and sampling strategy '{strategy}'"
            )
            print(f"  Searched pattern: {pattern}")

    return strategy_paths


def main(
    base_path: str, model: str, test_imbalance: str, sampling_strategies: list[str]
):
    # Find sampling strategy directories
    strategy_paths = find_sampling_strategy_directories(
        base_path, model, test_imbalance, sampling_strategies
    )

    if not strategy_paths:
        print("No sampling strategy directories found!")
        return

    print(f"Found strategies: {list(strategy_paths.keys())}")

    # Create save directory using model, test imbalance and comparison type
    save_dir = f"{base_path}/figures/sampling_comparison/{model}/{test_imbalance}/"
    os.makedirs(save_dir, exist_ok=True)

    # Create subdirectories for different plot types
    os.makedirs(f"{save_dir}/raw", exist_ok=True)
    os.makedirs(f"{save_dir}/mse", exist_ok=True)
    os.makedirs(f"{save_dir}/improvement", exist_ok=True)

    # Collect metrics for each sampling strategy
    metric_stats = {}
    full_metrics = {}

    for strategy_name, eval_outputs_path in strategy_paths.items():
        print(f"Processing {strategy_name} from {eval_outputs_path}")
        try:
            metric_stats[strategy_name], full_metrics[strategy_name] = get_metric_stats(
                eval_outputs_path, plot=False
            )
        except Exception as e:
            print(f"Error processing {strategy_name}: {e}")
            continue

    if not metric_stats:
        print("No valid metrics found!")
        return

    # Generate plots using functions
    sampling_comparison_raw(
        save_dir=save_dir, metric_stats=metric_stats, full_metrics=full_metrics
    )
    sampling_comparison_mse(save_dir=save_dir, metric_stats=metric_stats)
    sampling_comparison_improvement(save_dir=save_dir, metric_stats=metric_stats)

    print(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compare different sampling strategies for the same model "
            "and test imbalance."
        )
    )
    parser.add_argument(
        "base_path",
        help="Base path like 'outputs/reddit_dataset_12/one-vs-all/football/42_05'",
    )
    parser.add_argument(
        "model",
        help="Model name (e.g., 'distilbert', 'gpt2', 'ModernBERT')",
    )
    parser.add_argument(
        "test_imbalance",
        help="Test imbalance level (e.g., '05', '01', '001')",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        required=True,
        help=(
            "List of sampling strategies to compare "
            "(e.g., 'random' 'distance' 'isolation')"
        ),
    )
    args = parser.parse_args()

    main(args.base_path, args.model, args.test_imbalance, args.strategies)
