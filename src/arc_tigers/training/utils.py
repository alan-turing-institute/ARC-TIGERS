from collections import Counter

import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from numpy.random import Generator
from torch import nn
from transformers import PreTrainedTokenizer, Trainer

from arc_tigers.constants import DATA_DIR
from arc_tigers.data.utils import (
    BINARY_COMBINATIONS,
    ONE_VS_ALL_COMBINATIONS,
    get_target_mapping,
    preprocess_function,
)
from arc_tigers.utils import get_device


def balance_dataset(dataset: Dataset, random_generator: Generator) -> Dataset:
    """
    This function takes a dataset and balances it by resamplling the classes to have an
    equal number of samples for each class. It also

    Args:
        dataset: Huggingface dataset to be balanced
        random_generator: Random generator to be used for shuffling the dataset

    Returns:
        dataset: A balanced dataset with an equal number of samples for each class
    """
    # Get the number of samples in each class
    label_counts = Counter(dataset["label"])
    # Calculate the number for each class
    min_samples = min(label_counts.values())
    resampled_data = []
    for label in label_counts:
        label_data = dataset.filter(lambda x, label=label: x["label"] == label)
        resampled_data.append(
            label_data.shuffle(generator=random_generator).select(range(min_samples))
        )
    return concatenate_datasets(resampled_data)


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        assert "loss_weights" in kwargs, "loss_weights must be provided"
        self.loss_weights = kwargs.pop("loss_weights", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(self.loss_weights).to(get_device())
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def get_reddit_data(
    setting: str,
    target_config: str,
    balanced: bool,
    n_rows: int,
    tokenizer: PreTrainedTokenizer,
    random_seed: int,
):
    """
    Loads and preprocesses the Reddit dataset based on the specified configuration.

    Args:
        setting (str): The classification setting, either "multi-class" or "one-vs-all".
        target_config (str): The target configuration to use for the dataset.
        balanced (bool): Whether to balance the dataset by resampling classes.
        n_rows (int): The number of rows to load from the dataset.
        tokenizer (PreTrainedTokenizer): The tokenizer to use for
        reprocessing.
        random_seed (int): The random seed to use for reproducibility.

    Raises:
        ValueError: If an unknown setting is provided.

    Returns:
        tuple: A tuple containing the training dataset, evaluation dataset, and test
        dataset.
    """
    random_generator = np.random.default_rng(seed=random_seed)
    # work out the data directory
    data_dir = f"{DATA_DIR}/reddit_dataset_12/{n_rows}_rows/splits/{target_config}/"
    # load dataset
    dataset: dict[str, Dataset] = load_dataset(
        "csv",
        data_files={"train": f"{data_dir}/train.csv", "eval": f"{data_dir}/test.csv"},
    )
    meta_data = {}

    if setting == "multi-class":
        train_targets = BINARY_COMBINATIONS[target_config]["train"]
        train_dataset = dataset["train"].filter(
            lambda y: y in train_targets, input_columns=["label"]
        )
        train_target_map = get_target_mapping(setting, train_targets)
        meta_data["train_target_map"] = train_target_map

    elif setting == "one-vs-all":
        train_dataset = dataset["train"]
        train_targets = ONE_VS_ALL_COMBINATIONS[target_config]["train"]
        test_targets = ONE_VS_ALL_COMBINATIONS[target_config]["test"]
        train_target_map = get_target_mapping(setting, train_targets)
        test_target_map = get_target_mapping(setting, test_targets)
        meta_data["train_target_map"] = train_target_map
        meta_data["test_target_map"] = test_target_map

    else:
        err_msg = (
            f"Unknown setting: {setting}. Please use 'multi-class' or 'one-vs-all'."
        )
        raise ValueError(err_msg)

    tokenized_train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "targets": train_target_map,
        },
    )
    if setting == "one-vs-all":
        tokenized_test_dataset: Dataset = dataset["test"].map(
            preprocess_function,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "targets": test_target_map,
            },
        )
    elif setting == "multi-class":
        # use the train set so hat the classes are consistent
        tokenized_train_dataset, tokenized_test_dataset = (
            tokenized_train_dataset.train_test_split(
                test_size=0.5, generator=random_generator
            ).values()
        )

    # balance the dataset
    if balanced:
        print("Balancing the datasets...")
        tokenized_train_dataset = balance_dataset(
            tokenized_train_dataset, random_generator=random_generator
        )
        tokenized_test_dataset = balance_dataset(
            tokenized_test_dataset, random_generator=random_generator
        )

    # Split dataset
    train_data, eval_data = tokenized_train_dataset.train_test_split(
        test_size=0.1, generator=random_generator
    ).values()
    return train_data, eval_data, tokenized_test_dataset, meta_data


def get_label_weights(dataset, verbose=True):
    label_counter = Counter(dataset["label"])
    label_counts = dict(sorted(label_counter.items(), key=lambda item: item[1]))
    label_weights = [
        1 - (label_count / sum(label_counts.values()))
        for label_count in label_counts.values()
    ]

    if verbose:
        print("Label counts in the training dataset:")
        for label_idx, (label, count) in enumerate(label_counts.items()):
            print(f"Label: {label}, Count: {count}, Weight: {label_weights[label_idx]}")

    return label_weights
