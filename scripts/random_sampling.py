import argparse
import json
import os
from copy import deepcopy

import numpy as np
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from arc_tigers.eval.utils import compute_metrics
from arc_tigers.sample.random import RandomSampler
from arc_tigers.training.utils import get_reddit_data
from arc_tigers.utils import load_yaml


def imbalance_dataset(dataset, seed, class_balance):
    # Imbalance the dataset based on the class_balance variable
    class_labels = np.array(dataset["label"])
    class_0_indices = np.where(class_labels == 0)[0]
    class_1_indices = np.where(class_labels == 1)[0]

    # Calculate the number of samples for each class based on class_balance
    n_class_0 = len(class_0_indices)
    n_class_1 = int(n_class_0 * class_balance)

    # Randomly sample indices for each class
    rng = np.random.default_rng(seed)
    sampled_class_0_indices = rng.choice(class_0_indices, n_class_0, replace=False)
    sampled_class_1_indices = rng.choice(class_1_indices, n_class_1, replace=False)

    # Combine the sampled indices and sort them
    sampled_indices = np.sort(
        np.concatenate([sampled_class_0_indices, sampled_class_1_indices])
    )

    # Subset the dataset and predictions
    dataset = dataset.select(sampled_indices)
    # Print class counts
    print(f"Class 0 count: {len(sampled_class_0_indices)}")
    print(f"Class 1 count: {len(sampled_class_1_indices)}")
    return dataset


def evaluate(dataset, preds) -> dict[str, float]:
    """
    Compute metrics for a given dataset with preds.

    Args:
        dataset: The dataset to compute metrics for
        preds: The predictions for the dataset.
        model: The model to compute metrics with.

    Returns:
        A dictionary containing the computed metrics.
    """
    # Placeholder for actual metric computation
    eval_pred = (preds, dataset["label"])
    metrics = compute_metrics(eval_pred)
    metric_names = list(metrics.keys())
    for key in metric_names:
        if isinstance(metrics[key], list):
            # unpack multi-valued metrics into separate values for each class
            multi_metric = metrics.pop(key)
            if len(multi_metric) == 1:
                msg = "If metric value is a list, should have more than 1 value"
                raise ValueError(msg)
            for i, m in enumerate(multi_metric):
                metrics[f"{key}_{i}"] = m

    return metrics


def sample_dataset_metrics(
    dataset: Dataset,
    preds,
    seed: int,
    max_labels: int | None = None,
    evaluate_steps: list[int] | None = None,
) -> list[dict[str, float]]:
    """
    Simulate iteratively random sampling the whole dataset, re-computing metrics
    after each sample.

    Args:
        dataset: The dataset to sample from.
        preds: The predictions for the dataset.
        model: The model to compute metrics with.
        seed: The random seed for sampling.
        max_labels: The maximum number of labels to sample. If None, the whole dataset
            will be sampled.

    Returns:
        A list of dictionaries containing the computed metrics after each sample.
    """
    if max_labels is None:
        max_labels = len(dataset)
    if evaluate_steps is None:
        evaluate_steps = list(range(1, max_labels + 1))
    evaluate_steps = deepcopy(evaluate_steps)
    sampler = RandomSampler(dataset, seed)
    metrics = []
    next_eval_step = evaluate_steps.pop(0)
    for n in range(max_labels):
        sampler.sample()
        if (n + 1) == next_eval_step:
            metric = evaluate(
                dataset[sampler.labelled_idx], preds[sampler.labelled_idx]
            )
            metric["n"] = n + 1
            metrics.append(metric)
            if evaluate_steps:
                next_eval_step = evaluate_steps.pop(0)
            else:
                break

    return metrics


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
        output_dir = f"{save_dir}/new_random_sampling_outputs/"
    os.makedirs(output_dir, exist_ok=True)

    # full dataset stats
    metrics = evaluate(dataset, preds)
    print(metrics)
    with open(f"{output_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f)

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
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)

    # calculate predictions for whole dataset
    print(f"Loading model and tokenizer from {args.save_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.save_dir)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    _, _, test_dataset, meta_data = get_reddit_data(
        **data_config["data_args"], tokenizer=tokenizer
    )

    if args.class_balance != 1.0:
        test_dataset = imbalance_dataset(
            test_dataset, seed=args.seed, class_balance=args.class_balance
        )

    training_args = TrainingArguments(output_dir="tmp", per_device_eval_batch_size=16)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    preds = trainer.predict(test_dataset, metric_key_prefix="").predictions

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
