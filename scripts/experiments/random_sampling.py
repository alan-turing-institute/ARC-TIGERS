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
        help=(
            "path to the data config yaml file, or 'synthetic' to generate a "
            "synthetic dataset"
        ),
    )
    parser.add_argument(
        "model_config",
        help=(
            "path to the model config yaml file, or 'beta_model' to use a "
            "synthetic model. model_adv must be set if using beta_model."
        ),
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
    parser.add_argument(
        "--model_adv",
        type=float,
        default=3.0,
        help=(
            "Model advantage parameter used to parameterize the performance of the "
            "synthetic Beta models (use either this or error rates)."
        ),
    )
    parser.add_argument(
        "--pos_error_rate",
        type=float,
        required=False,
        help=(
            "Error rate for positive samples if using a synthetic model "
            "(use either this and neg_error_rate or model_adv)"
        ),
    )
    parser.add_argument(
        "--neg_error_rate",
        type=float,
        required=False,
        help=(
            "Error rate for negative samples if using a synthetic model "
            "(use either this and pos_error_rate or model_adv)"
        ),
    )
    parser.add_argument(
        "--synthetic_samples",
        type=int,
        default=10000,
        help="Number of samples to generate if using a synthetic dataset",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=50,
        help="Re-compute metrics every eval_every samples",
    )
    parser.add_argument(
        "--min_labels",
        type=int,
        default=10,
        help="Minimum number of labels to sample before computing metrics",
    )

    args = parser.parse_args()

    # only used if using a synthetic model or dataset
    synthetic_args = {
        "model_adv": args.model_adv,
        "synthetic_samples": args.synthetic_samples,
        "positive_error_rate": args.pos_error_rate,
        "negative_error_rate": args.neg_error_rate,
    }

    preds, test_dataset = get_preds(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        save_dir=args.save_dir,
        class_balance=args.class_balance,
        seed=args.seed,
        synthetic_args=synthetic_args,
    )

    main(
        args.save_dir,
        args.n_repeats,
        test_dataset,
        preds,
        args.seed,
        args.max_labels,
        evaluate_steps=np.arange(
            args.min_labels, args.max_labels, args.eval_every
        ).tolist(),
        class_balance=args.class_balance,
    )
