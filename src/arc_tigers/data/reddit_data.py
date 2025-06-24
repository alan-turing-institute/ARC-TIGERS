import glob
from typing import Any

import numpy as np
import pyarrow as pa
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

from arc_tigers.constants import DATA_DIR
from arc_tigers.data.utils import (
    balance_dataset,
    get_target_mapping,
    imbalance_binary_dataset,
    load_arrow_table,
    preprocess_function,
)
from arc_tigers.training.utils import get_label_weights

DATA_DRIFT_COMBINATIONS = {
    "sport": {
        "train": ["r/soccer", "r/Cricket"],
        "test": ["r/nfl", "NFLv2", "r/NBATalk", "r/nba"],
    },
    "football": {
        "train": ["r/soccer", "r/FantasyPL"],
        "test": ["r/coys", "r/reddevils", "r/LiverpoolFC"],
    },
    "american_football": {
        "train": ["r/nfl", "r/NFLv2"],
        "test": ["r/fantasyfootball"],
    },
    "ami": {"train": ["r/AmItheAsshole"], "test": ["r/AmIOverreacting"]},
    "news": {"train": ["r/news"], "test": ["r/Worldnews"]},
    "advice": {"train": ["r/AskReddit"], "test": ["r/AskMenAdvice", "r/Advice"]},
}

ONE_VS_ALL_COMBINATIONS = {
    "football": {
        "train": ["r/soccer", "r/FantasyPL", "r/coys", "r/reddevils", "r/LiverpoolFC"],
        "test": ["r/soccer", "r/FantasyPL", "r/coys", "r/reddevils", "r/LiverpoolFC"],
    },
    "news": {"train": ["r/news", "r/Worldnews"], "test": ["r/news", "r/Worldnews"]},
    "advice": {
        "train": ["r/AskReddit", "r/AskMenAdvice", "r/Advice"],
        "test": ["r/AskReddit", "r/AskMenAdvice", "r/Advice"],
    },
}

BINARY_COMBINATIONS = {
    "sport": {
        "train": ["r/soccer", "r/Cricket"],
        "test": ["r/soccer", "r/Cricket"],
    },
    "football": {
        "train": ["r/soccer", "r/FantasyPL"],
        "test": ["r/soccer", "r/FantasyPL"],
    },
}


def load_data(data_dir) -> DatasetDict:
    # Check for shards
    train_shards = sorted(glob.glob(f"{data_dir}/train_shard_*.csv"))
    test_shards = sorted(glob.glob(f"{data_dir}/test_shard_*.csv"))

    # if shards are found, load them
    if train_shards and test_shards:
        print(
            f"Found {len(train_shards)} train shards "
            f"and {len(test_shards)} test shards."
        )
        print("loading train shards...")
        train_table = load_arrow_table(train_shards)
        print("loading test shards...")
        test_table = load_arrow_table(test_shards)
        return DatasetDict(
            {
                "train": Dataset(pa.Table.from_batches(train_table.to_batches())),
                "test": Dataset(pa.Table.from_batches(test_table.to_batches())),
            }
        )
    # otherwise load the full csv files
    return load_dataset(
        "csv",
        data_files={
            "train": f"{data_dir}/train.csv",
            "test": f"{data_dir}/test.csv",
        },
    )


def get_reddit_data(
    setting: str,
    target_config: str,
    balanced: bool,
    n_rows: int,
    tokenizer: PreTrainedTokenizer | None,
    class_balance: float | None = None,
    all_classes: bool = False,
) -> tuple[Dataset, Dataset, Dataset, dict[str, dict[str, Any]]]:
    """
    Loads and preprocesses the Reddit dataset based on the specified configuration.

    Args:
        setting (str): The classification setting, either "multi-class" , "one-vs-all",
        or "data-drift". target_config (str): The target configuration to use for the
        dataset.
        balanced (bool): Whether to balance the dataset by resampling classes.
        n_rows (int): The number of rows to load from the dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for
        reprocessing.
        random_seed (int): The random seed for reproducibility.
        class_balance (float | None): Typically used for evaluation stage. The class
            balance to use for imbalanced sampling.
            If None, the dataset will be left in its natural state.
        all_classes (bool): If True, all classes will be used for training. If False,
            only the classes specified in the target_config will be used. This is only
            used in the multi-class setting.

    Raises:
        ValueError: If an unknown setting is provided.

    Returns:
        tuple: A tuple containing the training dataset, evaluation dataset, and test
        dataset.
    """
    # This needs to be reproducible across all experiments
    # so hardcode seed in this case
    split_generator = np.random.default_rng(seed=42)

    data_dir_mode = "data-drift" if setting == "multi-class" else setting

    data_dir = (
        f"{DATA_DIR}/reddit_dataset_12/{n_rows}_rows/"
        f"splits/{target_config}_{data_dir_mode}/"
    )

    print(f"Loading data from {data_dir}...")

    dataset = load_data(data_dir)

    meta_data = {}

    if setting == "multi-class":
        if target_config not in BINARY_COMBINATIONS:
            err_msg = (
                f"Unknown target config: {target_config}. Please use one of "
                f"{list(BINARY_COMBINATIONS.keys())} or use 'all' to use all classes."
            )
            raise ValueError(err_msg)

        # allow for all classes to be used
        if all_classes:
            train_targets = dataset["train"]["label"]
            # remove duplicates
            train_targets = list(set(train_targets))

        else:
            train_targets = BINARY_COMBINATIONS[target_config]["train"]

        train_dataset = dataset["train"].filter(
            lambda y: y in train_targets, input_columns=["label"]
        )
        train_target_map = get_target_mapping(setting, train_targets)
        meta_data["train_target_map"] = train_target_map
        meta_data["test_target_map"] = train_target_map

    elif setting == "one-vs-all":
        if target_config not in ONE_VS_ALL_COMBINATIONS:
            err_msg = (
                f"Unknown target config: {target_config}. Please use one of "
                f"{list(ONE_VS_ALL_COMBINATIONS.keys())}."
            )
            raise ValueError(err_msg)

        train_targets = ONE_VS_ALL_COMBINATIONS[target_config]["train"]
        test_targets = ONE_VS_ALL_COMBINATIONS[target_config]["test"]
        train_dataset = dataset["train"]
        train_target_map = get_target_mapping(setting, train_targets)
        test_target_map = get_target_mapping(setting, test_targets)
        meta_data["train_target_map"] = train_target_map
        meta_data["test_target_map"] = train_target_map

    elif setting == "data-drift":
        if target_config not in DATA_DRIFT_COMBINATIONS:
            err_msg = (
                f"Unknown target config: {target_config}. Please use one of "
                f"{list(DATA_DRIFT_COMBINATIONS.keys())}."
            )
            raise ValueError(err_msg)
        train_dataset = dataset["train"]
        train_targets = DATA_DRIFT_COMBINATIONS[target_config]["train"]
        test_targets = DATA_DRIFT_COMBINATIONS[target_config]["test"]
        train_target_map = get_target_mapping(setting, train_targets)
        test_target_map = get_target_mapping(setting, test_targets)
        meta_data["train_target_map"] = train_target_map
        meta_data["test_target_map"] = test_target_map

    else:
        err_msg = (
            f"Unknown setting: {setting}. "
            "Please use 'multi-class' 'one-vs-all', or 'data-drift'."
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
    # If the dataset is multi-class, we split the train set into train and test sets
    tokenized_test_dataset: Dataset
    if setting == "multi-class":
        # use the train set so hat the classes are consistent between train and test
        tokenized_train_dataset, tokenized_test_dataset = (
            tokenized_train_dataset.train_test_split(
                test_size=0.5, generator=split_generator
            ).values()
        )
    else:
        tokenized_test_dataset = dataset["test"].map(
            preprocess_function,
            batched=True,
            fn_kwargs={
                "tokenizer": tokenizer,
                "targets": test_target_map,
            },
        )

    # balance the dataset
    if balanced:
        print("Balancing the datasets...")
        tokenized_train_dataset = balance_dataset(
            tokenized_train_dataset, split_generator
        )
        tokenized_test_dataset = balance_dataset(
            tokenized_test_dataset, split_generator
        )
    else:
        # perform imbalanced sampling if class_balance is not None:
        if class_balance:
            print(f"Imbalancing the dataset with a class balance of {class_balance}...")
            print("Training dataset:")
            tokenized_train_dataset = imbalance_binary_dataset(
                tokenized_train_dataset,
                class_balance=class_balance,
                generator=split_generator,
            )
            print("Test dataset:")
            tokenized_test_dataset = imbalance_binary_dataset(
                tokenized_test_dataset,
                class_balance=class_balance,
                generator=split_generator,
            )
        # otherwise leave natural imbalance

    # Split dataset
    train_data, eval_data = tokenized_train_dataset.train_test_split(
        test_size=0.1, generator=split_generator
    ).values()

    meta_data["train_label_counts"] = get_label_weights(train_data, verbose=False)[1]
    meta_data["test_label_counts"] = get_label_weights(
        tokenized_test_dataset, verbose=False
    )[1]
    meta_data["eval_label_counts"] = get_label_weights(eval_data, verbose=False)[1]

    return train_data, eval_data, tokenized_test_dataset, meta_data
