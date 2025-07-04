from datasets import Dataset, load_from_disk
from transformers.tokenization_utils import PreTrainedTokenizer

from arc_tigers.constants import CONFIG_DIR, DATA_DIR
from arc_tigers.data.utils import (
    hf_train_test_split,
    preprocess_function,
)
from arc_tigers.utils import load_yaml


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
