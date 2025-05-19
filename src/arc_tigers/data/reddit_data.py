import numpy as np
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

from arc_tigers.constants import DATA_DIR
from arc_tigers.data.utils import (
    balance_dataset,
    get_target_mapping,
    imbalance_binary_dataset,
    preprocess_function,
)
from arc_tigers.training.utils import get_label_weights

ONE_VS_ALL_COMBINATIONS = {
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


def get_reddit_data(
    setting: str,
    target_config: str,
    balanced: bool,
    n_rows: int,
    tokenizer: PreTrainedTokenizer | None,
    random_seed: int,
    class_balance: float | None = None,
):
    """
    Loads and preprocesses the Reddit dataset based on the specified configuration.

    Args:
        setting (str): The classification setting, either "multi-class" or "one-vs-all".
        target_config (str): The target configuration to use for the dataset.
        balanced (bool): Whether to balance the dataset by resampling classes.
        n_rows (int): The number of rows to load from the dataset.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use for
        reprocessing.
        random_seed (int): The random seed for reproducibility.
        class_balance (float | None): Typically used for evaluation stage. The class
            balance to use for imbalanced sampling.
            If None, the dataset will be left in its natural state.

    Raises:
        ValueError: If an unknown setting is provided.

    Returns:
        tuple: A tuple containing the training dataset, evaluation dataset, and test
        dataset.
    """
    split_generator = np.random.default_rng(seed=random_seed)
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
        meta_data["test_target_map"] = train_target_map

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
                test_size=0.5, generator=split_generator
            ).values()
        )

    # balance the dataset
    if balanced:
        print("Balancing the datasets...")
        tokenized_train_dataset = balance_dataset(tokenized_train_dataset)
        tokenized_test_dataset = balance_dataset(tokenized_test_dataset)
    else:
        # perform imbalanced sampling if class_balance is not None:
        if class_balance:
            print(f"Imbalancing the dataset with a class balance of {class_balance}...")
            print("Training dataset:")
            tokenized_train_dataset = imbalance_binary_dataset(
                tokenized_train_dataset,
                seed=random_seed,
                class_balance=class_balance,
            )
            print("Test dataset:")
            tokenized_test_dataset = imbalance_binary_dataset(
                tokenized_test_dataset,
                seed=random_seed,
                class_balance=class_balance,
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
