from collections import Counter

import torch
from datasets import concatenate_datasets, load_dataset
from torch import nn
from transformers import Trainer

from arc_tigers.constants import DATA_DIR
from arc_tigers.data.utils import (
    BINARY_COMBINATIONS,
    ONE_VS_ALL_COMBINATIONS,
    get_target_mapping,
    preprocess_function,
)
from arc_tigers.utils import get_device


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


def get_reddit_data(setting, target_config, balanced, n_rows, tokenizer):
    # work out the data directory
    data_dir = f"{DATA_DIR}/reddit_dataset_12/{n_rows}_rows/splits/{target_config}/"
    # load dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": f"{data_dir}/train.csv", "eval": f"{data_dir}/test.csv"},
    )

    if setting == "multi-class":
        targets = BINARY_COMBINATIONS[target_config]["train"]
        targets = ["r/soccer", "r/FantasyPL"]
        dataset = dataset.filter(lambda y: y in targets, input_columns=["label"])
        target_map = get_target_mapping(setting, targets)

    elif setting == "one-vs-all":
        targets = ONE_VS_ALL_COMBINATIONS[target_config]["train"]
        target_map = get_target_mapping(setting, targets)

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "targets": target_map,
        },
    )

    # balance the dataset
    if balanced:
        print("Balancing the dataset...")
        # Get the number of samples in each class
        label_counts = Counter(tokenized_datasets["train"]["label"])
        # Calculate the weights for each class
        min_samples = min(label_counts.values())
        resampled_data = []
        for label in label_counts:
            label_data = tokenized_datasets["train"].filter(
                lambda x, label=label: x["label"] == label
            )
            resampled_data.append(label_data.shuffle().select(range(min_samples)))
        tokenized_datasets["train"] = concatenate_datasets(resampled_data)

    # Split dataset
    train_data = tokenized_datasets["train"]
    return train_data.train_test_split(test_size=0.1).values()


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
