import json
import os

import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    TrainingArguments,
)

from arc_tigers.sample.random import RandomSampler
from arc_tigers.training.utils import (
    WeightedLossTrainer,
    get_label_weights,
    get_reddit_data,
)


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
    """
    Iteratively sample a dataset and compute metrics for the labelled subset.

    Args:
        save_dir: Directory to save the metrics files.
        n_repeats: Number of times to repeat the sampling.
        dataset: The dataset to sample from.
        model: The model to compute metrics with.
        init_seed: The initial seed for random sampling (determines the seed used for
            each repeat).
        max_labels: The maximum number of labels to sample. If None, the whole dataset
            will be sampled.
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(init_seed)

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
    # calculate predictions for whole dataset
    model_name = model_config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_config["model_kwargs"]
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset, eval_dataset, _ = get_reddit_data(
        **data_config["data_args"], tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{save_dir}/logs",
        save_total_limit=3,
        load_best_model_at_end=True,
    )
    # Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        loss_weights=get_label_weights(train_dataset),
    )

    preds = ...  # Compute predictions
    # dataset["preds"] = preds

    main("data/random_sampling", 1000, np.array(range(10000)), None, 42, 500)
