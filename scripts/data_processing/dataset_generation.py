import os

from datasets import Dataset, DatasetDict, concatenate_datasets

from arc_tigers.constants import DATA_DIR
from arc_tigers.data.reddit_data import DATA_DRIFT_COMBINATIONS, ONE_VS_ALL_COMBINATIONS


def process_data(
    data: Dataset,
    target_categories,
    mode,
    train_imbalance,
    test_imbalance,
    save_path,
    seed,
    sort_by_length=True,
):
    """
    Processes a list of reddit data rows and generates train and test splits
    based on the provided target categories.

    This function can be used for both single large JSON files and individual shards.
    It separates rows into train and test sets according to the subreddit, applies
    optional class imbalance, and saves the resulting splits as CSV files.

    Args:
        data (list): List of reddit data rows (dicts) to process.
        target_categories (dict): Dictionary with 'train' and 'test' subreddit lists.
        args (argparse.Namespace): Arguments including optional imbalance ratio.
        save_path (str): Directory where output CSVs will be saved.
        shard_id (int, optional): If processing a shard, the shard index (used in output
          filenames). If None, outputs are named 'train.csv' and 'test.csv'.
    """

    for split in ["train", "test"]:
        target_categories[split] = [x.lower() for x in target_categories[split]]

    data = data.select_columns(["text", "label"])

    def clean(row):
        row["text"] = row["text"].replace("\n", " ")
        row["len"] = len(row["text"])
        return row

    data = data.map(clean)

    # sort by length or shuffle
    data = data.sort("len", reverse=True) if sort_by_length else data.shuffle(seed=seed)

    if mode == "one-vs-all":
        targets = data.filter(lambda x: x["label"] in target_categories["train"])
        target_split = targets.train_test_split(test_size=0.5, seed=seed)
        train_targets = target_split["train"]
        test_targets = target_split["test"]
        non_targets = data.filter(
            lambda x: x["label"] not in target_categories["train"]
        )
    else:
        train_targets = data.filter(lambda x: x["label"] in target_categories["train"])
        test_targets = data.filter(lambda x: x["label"] in target_categories["test"])
        non_targets = data.filter(
            lambda x: x["label"] not in target_categories["train"]
            and x["label"] not in target_categories["test"]
        )

    # Imbalance the dataset
    if train_imbalance is not None:
        n_train_non_targets = int(len(train_targets) * (1 / train_imbalance - 1))
    else:
        n_train_non_targets = int(
            len(non_targets)
            * (len(train_targets) / (len(train_targets) + len(test_targets)))
        )
    if test_imbalance is not None:
        n_test_non_targets = int(len(test_targets) * (1 / test_imbalance - 1))
    else:
        n_test_non_targets = int(
            len(non_targets)
            * (len(test_targets) / (len(train_targets) + len(test_targets)))
        )

    print(
        f"Train targets: {len(train_targets)} | "
        f"Test targets: {len(test_targets)} | "
        f"Train non-targets: {n_train_non_targets} | "
        f"Test non-targets: {n_test_non_targets} |"
        f"Total non-targets: {len(non_targets)}"
    )

    if n_train_non_targets + n_test_non_targets > len(non_targets):
        msg = (
            "The requested imbalance ratio is too high, not enough non-target samples "
            "available."
        )
        raise ValueError(msg)

    non_target_split = non_targets.train_test_split(
        test_size=n_train_non_targets, seed=seed
    )
    non_targets = non_target_split["train"]
    train_non_targets = non_target_split["test"]
    print(train_targets)
    print(train_non_targets)

    non_target_split = non_targets.train_test_split(
        test_size=n_test_non_targets, seed=seed
    )
    non_targets = non_target_split["train"]
    test_non_targets = non_target_split["test"]

    # concatenate targets with non-targets
    train_data = concatenate_datasets([train_targets, train_non_targets]).shuffle(
        seed=seed
    )
    test_data = concatenate_datasets([test_targets, test_non_targets]).shuffle(
        seed=seed
    )
    DatasetDict({"train": train_data, "test": test_data}).save_to_disk(save_path)
    print(f"Saved: {save_path}")


def main(args):
    ds = Dataset.load_from_disk(args.data_dir)

    if args.mode == "one-vs-all":
        target_categories = ONE_VS_ALL_COMBINATIONS[args.target_config]
    else:
        target_categories = DATA_DRIFT_COMBINATIONS[args.target_config]

    save_dir = (
        args.data_dir
        if os.path.isdir(args.data_dir)
        else os.path.dirname(args.data_dir)
    )
    save_path = f"{save_dir}/splits/{args.target_config}_{args.mode}/"
    os.makedirs(save_path, exist_ok=True)

    process_data(
        ds,
        target_categories,
        args.mode,
        args.train_imbalance,
        args.test_imbalance,
        save_path,
        args.seed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument(
        "data_dir",
        type=str,
        default=f"{DATA_DIR}/reddit_dataset_12",
        help="Path to the data file or directory containing shards",
    )
    parser.add_argument(
        "target_config",
        type=str,
        help=(
            "Split to generate, options are: sport, american_football, ami, news, "
            "advice"
        ),
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="one-vs-all",
        choices=["data-drift", "one-vs-all", "multi-class"],
        help="Splitting mode: 'data-drift', 'one-vs-all, or 'multi-class''.",
    )
    parser.add_argument(
        "--train_imbalance",
        type=float,
        help=(
            "Imbalance ratio for the train dataset, 0.01 means 1% of the dataset is "
            "target classes."
        ),
    )
    parser.add_argument(
        "--test_imbalance",
        type=float,
        help=(
            "Imbalance ratio for the test dataset, 0.01 means 1% of the dataset is "
            "target classes."
        ),
    )
    parser.add_argument(
        "--seed", default=42, help="Random seed for splitting and sampling."
    )
    args = parser.parse_args()
    main(args)
