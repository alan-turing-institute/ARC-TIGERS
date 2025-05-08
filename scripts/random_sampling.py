import argparse
import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
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
    for key in metrics:
        if isinstance(metrics[key], np.ndarray):
            metrics[key] = metrics[key].tolist()
    return metrics


def sample_dataset_metrics(
    dataset: Dataset, preds, seed: int, max_labels: int | None = None
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
    sampler = RandomSampler(dataset, seed)
    metrics = []
    for i in range(max_labels):
        sampler.sample()
        metric = evaluate(dataset[sampler.labelled_idx], preds[sampler.labelled_idx])
        metric["n"] = i + 1
        metrics.append(metric)

    return metrics


def main(
    save_dir: str,
    n_repeats: int,
    dataset: Dataset,
    preds,
    init_seed: int,
    max_labels: int | None = None,
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
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(init_seed)

    # full dataset stats
    metrics = evaluate(dataset, preds)
    print(metrics)
    with open(f"{save_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f)

    # iteratively sample dataset and compute metrics, repeated n_repeats times
    for _ in range(n_repeats):
        seed = rng.integers(1, 2**32 - 1)  # Generate a random seed
        metrics = sample_dataset_metrics(dataset, preds, seed, max_labels=max_labels)
        pd.DataFrame(metrics).to_csv(f"{save_dir}/metrics_{seed}.csv", index=False)


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
    parser.add_argument("--n_repeats", type=int)
    parser.add_argument("--max_labels", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "save_dir",
        type=str,
        default=None,
        help="Path to save the model and results",
    )
    args = parser.parse_args()
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)

    # calculate predictions for whole dataset
    model_name = model_config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_config["model_kwargs"]
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    _, _, test_dataset = get_reddit_data(
        **data_config["data_args"], tokenizer=tokenizer
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

    main(args.save_dir, args.n_repeats, test_dataset, preds, args.seed, args.max_labels)
