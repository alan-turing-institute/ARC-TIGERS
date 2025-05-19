import argparse
import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm

from arc_tigers.data.utils import sample_dataset_metrics
from arc_tigers.eval.reddit_eval import get_preds
from arc_tigers.eval.utils import evaluate, get_stats


def main(
    save_dir: str,
    n_repeats: int,
    dataset: Dataset,
    preds,
    init_seed: int,
    max_labels: int | None = None,
    evaluate_steps: list[int] | None = None,
    class_balance: float = 1.0,
):
    """
    Iteratively sample a dataset and compute metrics for the labelled subset.

    Args:
        save_dir: Directory to save the metrics files.
        n_repeats: Number of times to repeat the sampling.
        dataset: The dataset to sample from
        preds: The predictions for the dataset.
        model: The model to compute metrics with.
        init_seed: The initial seed for random sampling (determines the seed used for
            each repeat).
        max_labels: The maximum number of labels to sample. If None, the whole dataset
            will be sampled.
    """

    rng = np.random.default_rng(init_seed)
    if class_balance != 1.0:
        output_dir = (
            f"{save_dir}/imbalanced_random_sampling_outputs_"
            f"{str(class_balance).replace('.', '')}/"
        )
    else:
        output_dir = f"{save_dir}/random_sampling_outputs/"
    os.makedirs(output_dir, exist_ok=True)

    # full dataset stats
    metrics = evaluate(dataset, preds)
    stats = get_stats(preds, dataset["label"])

    with open(f"{output_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{output_dir}/stats_full.json", "w") as f:
        json.dump(stats, f, indent=2)

    # iteratively sample dataset and compute metrics, repeated n_repeats times
    for _ in tqdm(range(n_repeats)):
        seed = rng.integers(1, 2**32 - 1)  # Generate a random seed
        metrics = sample_dataset_metrics(
            dataset, preds, seed, max_labels=max_labels, evaluate_steps=evaluate_steps
        )
        pd.DataFrame(metrics).to_csv(f"{output_dir}/metrics_{seed}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument(
        "data_config",
        help="path to the data config yaml file",
    )
    parser.add_argument(
        "model_config",
        help="path to the model config yaml file",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        default=None,
        help="Path to save the model and results",
    )
    parser.add_argument(
        "--class_balance",
        type=float,
        default=1.0,
        help="Balance between the classes",
    )
    parser.add_argument("--n_repeats", type=int, required=True)
    parser.add_argument("--max_labels", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    preds, test_dataset = get_preds(args)
    main(
        args.save_dir,
        args.n_repeats,
        test_dataset,
        preds,
        args.seed,
        args.max_labels,
        evaluate_steps=np.arange(5, args.max_labels, 10).tolist(),
        class_balance=args.class_balance,
    )
