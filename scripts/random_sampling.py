import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import PreTrainedModel

from arc_tigers.sample.random import RandomSampler


def compute_metrics(dataset, model) -> dict[str, float]:
    """
    Compute metrics for the given dataset and model.

    Args:
        dataset: The dataset to compute metrics for.
        model: The model to compute metrics with.

    Returns:
        A dictionary containing the computed metrics.
    """
    # Placeholder for actual metric computation
    return {
        "accuracy": np.mean(np.random.rand(len(dataset))),
        "f1_score": np.mean(np.random.rand(len(dataset))),
    }


def sample_dataset_metrics(
    dataset: Dataset, model: PreTrainedModel, seed: int, max_labels: int | None = None
) -> list[dict[str, float]]:
    """
    Simulate iteratively random sampling the whole dataset, re-computing metrics
    after each sample.

    Args:
        dataset: The dataset to sample from.
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
        metric = compute_metrics(dataset[sampler.labelled_idx], model)
        metric["n"] = i + 1
        metrics.append(metric)

    return metrics


def main(
    save_dir: str,
    n_repeats: int,
    dataset: Dataset,
    model: PreTrainedModel,
    init_seed: int,
    max_labels: int | None = None,
):
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(init_seed)

    # calculate predictions for whole dataset
    model = ...  # Load model
    preds = ...  # Compute predictions
    # dataset["preds"] = preds

    # full dataset stats
    metrics = compute_metrics(dataset, model)
    with open(f"{save_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f)

    # iteratively sample dataset and compute metrics, repeated n_repeats times
    for _ in range(n_repeats):
        seed = rng.integers(1, 2**32 - 1)  # Generate a random seed
        metrics = sample_dataset_metrics(dataset, model, seed, max_labels=max_labels)
        pd.DataFrame(metrics).to_csv(f"{save_dir}/metrics_{seed}.csv", index=False)


if __name__ == "__main__":
    main("data/random_sampling", 1000, np.array(range(10000)), None, 42, 500)
