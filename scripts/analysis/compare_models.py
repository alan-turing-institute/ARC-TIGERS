import argparse
import os

from arc_tigers.eval.plotting import model_comparison_mse, model_comparison_raw
from arc_tigers.eval.utils import find_model_directories, get_metric_stats


def main(base_path: str, models: list[str], test_imbalance: str, sampling_method: str):
    # Find model directories
    model_paths = find_model_directories(
        base_path, models, test_imbalance, sampling_method
    )

    if not model_paths:
        print("No model directories found!")
        return

    print(f"Found models: {list(model_paths.keys())}")

    # Create save directory using test imbalance and sampling method
    save_dir = (
        f"{base_path}/figures/model_comparison/{test_imbalance}/{sampling_method}/"
    )
    os.makedirs(save_dir, exist_ok=True)

    # Create subdirectories for different plot types
    os.makedirs(f"{save_dir}/raw", exist_ok=True)
    os.makedirs(f"{save_dir}/mse", exist_ok=True)

    # Collect metrics for each model
    metric_stats = {}
    full_metrics = {}

    for model_name, eval_outputs_path in model_paths.items():
        print(f"Processing {model_name} from {eval_outputs_path}")
        try:
            metric_stats[model_name], full_metrics[model_name] = get_metric_stats(
                eval_outputs_path, plot=False
            )
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue

    if not metric_stats:
        print("No valid metrics found!")
        return

    # Generate plots
    model_comparison_raw(
        save_dir=save_dir, metric_stats=metric_stats, full_metrics=full_metrics
    )
    model_comparison_mse(save_dir=save_dir, metric_stats=metric_stats)

    print(f"Plots saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare different models using the same sampling method."
    )
    parser.add_argument(
        "base_path",
        help="Base path like 'outputs/reddit_dataset_12/one-vs-all/football/42_05'",
    )
    parser.add_argument(
        "test_imbalance",
        help="Test imbalance level (e.g., '05', '01', '001')",
    )
    parser.add_argument(
        "sampling_method",
        help="Sampling method directory name (e.g., 'random', 'distance')",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of model names to compare (e.g., 'gpt2' 'distilbert-base-uncased')",
    )
    args = parser.parse_args()

    main(args.base_path, args.models, args.test_imbalance, args.sampling_method)
