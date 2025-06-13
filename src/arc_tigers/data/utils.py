from collections import Counter
from collections.abc import Generator
from copy import deepcopy
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.csv as pv
from datasets import Dataset, concatenate_datasets
from numpy.random import BitGenerator
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from arc_tigers.eval.utils import evaluate
from arc_tigers.sample.acquisition import AcquisitionFunction, BiasCorrector

EXPECTED_KEYS = {"text", "label", "len"}


def is_valid_row(row):
    # Check the row is a dictionary and contains the expected keys
    return isinstance(row, dict) and EXPECTED_KEYS.issubset(row.keys())


def load_arrow_table(shard_files):
    parse_options = pv.ParseOptions(newlines_in_values=True)
    # Read all shards as Arrow tables and concatenate
    tables = []
    for table_index, f in enumerate(shard_files):
        print(f"Loading shard {table_index + 1}/{len(shard_files)}: {f}")
        tables.append(pv.read_csv(f, parse_options=parse_options))
    return pa.concat_tables(tables)


def imbalance_binary_dataset(
    dataset: Dataset,
    class_balance: float,
    generator: BitGenerator,
) -> Dataset:
    """
    Imbalance the dataset based on the class_balance variable.

    Args:
        dataset: The dataset to imbalance.
        seed: The random seed for sampling.
        class_balance: The balance between the classes. A value of 1.0 means
            balanced classes, while a value of 0.5 means class 1 is half the size of
            class 0. A negative value means class 1 is larger than class 0.

    Returns:
        The imbalanced dataset.
    """
    # Imbalance the dataset based on the class_balance variable
    class_labels = np.array(dataset["label"])
    class_0_indices = np.where(class_labels == 0)[0]
    class_1_indices = np.where(class_labels == 1)[0]

    # Calculate the number of samples for each class based on class_balance
    assert abs(class_balance) <= 1.0, "class balance must be between -1.0 and 1.0"
    if class_balance < 0:
        class_balance = -1.0 * class_balance
        n_class_1 = len(class_1_indices)
        n_class_0 = int(n_class_1 * class_balance)
        if n_class_0 >= len(class_0_indices):
            err_msg = "class balance is too large for class 0"
            raise ValueError(err_msg)
    else:
        n_class_0 = len(class_0_indices)
        n_class_1 = int(n_class_0 * class_balance)
        if n_class_1 >= len(class_1_indices):
            err_msg = "class balance is too large for class 0"
            raise ValueError(err_msg)

    # Randomly sample indices for each class
    sampled_class_0_indices = generator.choice(
        class_0_indices, n_class_0, replace=False
    )
    sampled_class_1_indices = generator.choice(
        class_1_indices, n_class_1, replace=False
    )

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


def flag_row(row: dict) -> bool:
    """
    Flags a row for removal if it is empty, deleted, or removed. It also flags
    rows that are too short (less than 10 characters). It also flags rows
    that are posts and not comments.

    Args:
        row: row from the reddit dataset

    Returns:
        bool True if the row should be flagged for removal, False otherwise
    """
    if not row["text"]:
        return True
    if row["text"] == "[deleted]":
        return True
    if row["text"] == "[removed]":
        return True
    if len(row["text"]) < 10:
        return True
    return row["dataType"] == "post"


def balance_dataset(dataset: Dataset, split_generator: Generator) -> Dataset:
    # Get the number of samples in each class
    label_counts = Counter(dataset["label"])
    # Calculate the number for each class
    min_samples = min(label_counts.values())
    resampled_data = []
    for label in label_counts:
        label_data = dataset.filter(lambda x, label=label: x["label"] == label)
        resampled_data.append(
            label_data.shuffle(generator=split_generator).select(range(min_samples))
        )
    return concatenate_datasets(resampled_data)


def clean_row(row: dict) -> dict:
    """
    The function takse a row of the reddit dataset with all of the fields. It then
    removes the extra fields and returns a new row with only the relevant fields
    (text, label, and the length). It also renames these appropriately.
    Args:
        row: row from the reddit dataset

    Returns:
        new_row: a new row with only the relevant fields
    """
    clean_text = row["text"].replace("\n", " ")
    new_row = {}
    new_row["text"] = clean_text
    new_row["label"] = row["communityName"]
    new_row["len"] = len(row["text"])
    return new_row


def get_target_mapping(
    eval_setting: str, target_subreddits: list[str]
) -> dict[str, int]:
    if eval_setting == "multi-class":
        return {subreddit: index for index, subreddit in enumerate(target_subreddits)}
    if eval_setting == "one-vs-all":
        return dict.fromkeys(target_subreddits, 1)
    err_msg = f"Invalid evaluation setting: {eval_setting}"
    raise ValueError(err_msg)


def preprocess_function(
    examples: dict[str, Any],
    tokenizer: PreTrainedTokenizer | None,
    targets: dict[str, int],
) -> dict[str, Any]:
    if tokenizer:
        tokenized = tokenizer(examples["text"], padding=True, truncation=True)
    else:
        tokenized = examples
    tokenized["label"] = [targets.get(label, 0) for label in examples["label"]]
    return tokenized


def sample_dataset_metrics(
    dataset: Dataset,
    preds: np.ndarray,
    sampler: AcquisitionFunction,
    max_labels: int | None = None,
    evaluate_steps: list[int] | None = None,
    bias_corrector: BiasCorrector | None = None,
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
    metrics = []
    next_eval_step = evaluate_steps.pop(0)
    for n in tqdm(range(max_labels)):
        q = sampler.sample()
        if (n + 1) == next_eval_step:
            metric = evaluate(
                dataset[sampler.labelled_idx], preds[sampler.labelled_idx]
            )
            if bias_corrector:
                metric = bias_corrector.apply_weighting_to_dict(
                    q=q, m=n, metrics=metric
                )
            metric["n"] = n + 1
            metrics.append(metric)
            if evaluate_steps:
                next_eval_step = evaluate_steps.pop(0)
            else:
                break

    return metrics
