import argparse
import os
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets

from arc_tigers.constants import DATA_DIR
from arc_tigers.data.reddit_data import DATA_DRIFT_COMBINATIONS, ONE_VS_ALL_COMBINATIONS
from arc_tigers.data.utils import hf_train_test_split
from arc_tigers.utils import load_yaml


def n_non_targets_from_imbalance(
    n_targets_split: int,
    imbalance_ratio: float | None,
    total_non_targets: int,
    total_targets: int,
) -> int:
    """
    Computes the number of non-target samples needed to achieve the specified imbalance
    ratio in a dataset with a given number of target samples and total non-target
    samples.

    Args:
        n_targets (int): Number of target samples.
        imbalance_ratio (float): Desired imbalance ratio (e.g., 0.01 for 1%).
        total_non_targets (int): Total number of non-target samples available.

    Returns:
        int: Number of non-target samples needed.
    """
    if imbalance_ratio is None:  # preserve natural imbalance
        return int(total_non_targets * (n_targets_split / total_targets))

    # calculate the number of non-targets needed to achieve the imbalance ratio
    return int(n_targets_split * (1 / imbalance_ratio - 1))


def process_data(
    data: Dataset,
    target_categories: dict[str, list[str]],
    mode: str,
    train_imbalance: float | None,
    test_imbalance: float | None,
    save_path: str,
    seed: int,
):
    """
    Processes the dataset to create train and test splits based on target categories
    (lists of subreddit names) and specified imbalance ratios, and saves the processed
    dataset to the specified path.

    Args:
        data: The dataset to process.
        target_categories: Dictionary with keys "train" and "test" containing lists of
            target categories (subreddit names) for each split.
        mode: The evaluation setting, either "one-vs-all" or "data-drift", or "binary"
        train_imbalance: The desired imbalance ratio for the training set, or None to
            preserve the natural imbalance.
        test_imbalance: The desired imbalance ratio for the test set, or None to
            preserve the natural imbalance.
        save_path: The path where the processed dataset will be saved.
        seed: Random seed for reproducibility.
    """
    for split in ["train", "test"]:
        target_categories[split] = [x.lower() for x in target_categories[split]]

    data = data.select_columns(["text", "label"])
    data = data.rename_column("label", "subreddit")

    def clean(row):
        row["text"] = row["text"].replace("\n", " ")
        row["len"] = len(row["text"])
        return row

    data = data.map(clean)
    data = data.shuffle(seed=seed)

    if mode == "one-vs-all":
        # target categories defines the positive class for both the train and test sets
        targets = data.filter(lambda x: x["subreddit"] in target_categories["train"])
        train_targets, test_targets = hf_train_test_split(
            targets, test_size=0.5, seed=seed
        )
        non_targets = data.filter(
            lambda x: x["subreddit"] not in target_categories["train"]
        )
    else:
        # target categories defines the positive class for train and test separately
        train_targets = data.filter(
            lambda x: x["subreddit"] in target_categories["train"]
        )
        test_targets = data.filter(
            lambda x: x["subreddit"] in target_categories["test"]
        )
        non_targets = data.filter(
            lambda x: x["subreddit"] not in target_categories["train"]
            and x["subreddit"] not in target_categories["test"]
        )

    # Compute the number of non-target samples for train and test sets, either using the
    # requested imbalance ratio or preserving the natural imbalance in the dataset.
    n_train_targets = len(train_targets)
    n_test_targets = len(test_targets)
    n_total_targets = n_train_targets + n_test_targets
    n_total_non_targets = len(non_targets)

    n_train_non_targets = n_non_targets_from_imbalance(
        n_train_targets, train_imbalance, n_total_non_targets, n_total_targets
    )
    n_test_non_targets = n_non_targets_from_imbalance(
        n_test_targets, test_imbalance, n_total_non_targets, n_total_targets
    )
    print(
        f"Train targets: {n_train_targets} | "
        f"Test targets: {n_test_targets} | "
        f"Train non-targets: {n_train_non_targets} | "
        f"Test non-targets: {n_test_non_targets} |"
        f"Total non-targets: {n_test_non_targets}"
    )
    if n_train_non_targets + n_test_non_targets > len(non_targets):
        msg = (
            "The requested imbalance ratio is too high, not enough non-target samples "
            "available."
        )
        raise ValueError(msg)

    # Get required number of non-target samples for train and test sets
    non_targets, train_non_targets = hf_train_test_split(
        non_targets, test_size=n_train_non_targets, seed=seed
    )
    non_targets, test_non_targets = hf_train_test_split(
        non_targets, test_size=n_test_non_targets, seed=seed
    )

    # add class label
    train_targets = train_targets.add_column(name="label", column=[1] * n_train_targets)
    train_non_targets = train_non_targets.add_column(
        name="label", column=[0] * n_train_non_targets
    )
    test_targets = test_targets.add_column(name="label", column=[1] * n_test_targets)
    test_non_targets = test_non_targets.add_column(
        name="label", column=[0] * n_test_non_targets
    )

    # concatenate targets with non-targets
    train_data = concatenate_datasets([train_targets, train_non_targets]).shuffle(
        seed=seed
    )
    test_data = concatenate_datasets([test_targets, test_non_targets]).shuffle(
        seed=seed
    )
    DatasetDict({"train": train_data, "test": test_data}).save_to_disk(save_path)
    print(f"Saved: {save_path}")


def main(config_path: str):
    config = load_yaml(config_path)

    data_dir = f"{DATA_DIR}/{config['data_name']}"
    ds = Dataset.load_from_disk(data_dir)

    if config["setting"] == "one-vs-all":
        target_categories = ONE_VS_ALL_COMBINATIONS[config["target_config"]]
    else:
        target_categories = DATA_DRIFT_COMBINATIONS[config["target_config"]]

    save_name = Path(config_path).name.rstrip(".yaml")
    save_dir = f"{data_dir}/splits/{save_name}"
    os.makedirs(save_dir, exist_ok=True)

    process_data(
        ds,
        target_categories,
        config["setting"],
        config["train_imbalance"],
        config["test_imbalance"],
        save_dir,
        config["seed"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("data_config", type=str, help="Path to the data config file")
    args = parser.parse_args()
    main(args.data_config)
