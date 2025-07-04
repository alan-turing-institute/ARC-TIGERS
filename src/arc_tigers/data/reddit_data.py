import glob

import pyarrow as pa
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers.tokenization_utils import PreTrainedTokenizer

from arc_tigers.constants import CONFIG_DIR, DATA_DIR
from arc_tigers.data.utils import (
    hf_train_test_split,
    load_arrow_table,
    preprocess_function,
)
from arc_tigers.utils import load_yaml

# DATA_DRIFT_COMBINATIONS specifies settings for "data-drift" experiments.
# In this setting, we perform one-vs-all classification, but the set of target classes
# changes between the train and test splits. Each split has its own exclusive set of
# target subreddits (as defined in the dictionary), and there is no overlap between
# the train and test target classes. This allows us to evaluate model generalization
# to new, unseen target classes under distribution shift.
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

# ONE_VS_ALL_COMBINATIONS specifies settings for "one-vs-all" experiments.
# In this setting, we perform one-vs-all classification, where the same set of target
# subreddits (as defined in the dictionary) are used for both the train and test splits.
# All samples from these target subreddits are considered positive examples, and all
# other subreddits are considered negative examples. This allows us to evaluate
# performance when the model is trained and tested on the same set of target classes.
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

# BINARY_COMBINATIONS specifies settings for standard binary classification experiments.
# In this setting, only the specified two target subreddits (or classes) are used for
# both the train and test splits. All samples from these subreddits are included, and
# the task is to distinguish between them (i.e., no "other" or negative class is
# present). This allows for straightforward binary classification between two specific
# classes.
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
    data_config_name: str,
    tokenizer: PreTrainedTokenizer | None,
) -> tuple[Dataset, Dataset, Dataset]:
    """
    Loads and preprocesses the Reddit dataset based on the specified configuration.

    Returns:
        tuple: A tuple containing the training dataset, evaluation dataset, and test
        dataset.
    """
    data_config_path = f"{CONFIG_DIR}/data/{data_config_name}.yaml"
    data_config = load_yaml(data_config_path)
    data_dir = f"{DATA_DIR}/{data_config['data_name']}/splits/{data_config_name}"

    print(f"Loading data from {data_dir}...")
    dataset = load_from_disk(data_dir)

    tokenized_train_dataset = dataset["train"].map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )
    tokenized_test_dataset = dataset["test"].map(
        preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
    )

    # Split dataset
    train_data, eval_data = hf_train_test_split(
        tokenized_train_dataset, test_size=0.1, seed=data_config["seed"]
    )

    return train_data, eval_data, tokenized_test_dataset
