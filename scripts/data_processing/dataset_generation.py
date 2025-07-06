import argparse
import logging
import os
from multiprocessing import cpu_count

from datasets import Dataset, DatasetDict, concatenate_datasets

from arc_tigers.data.config import HFDataConfig
from arc_tigers.data.utils import hf_train_test_split

logger = logging.getLogger(__name__)

num_proc = min(cpu_count(), 8)


def get_full_splits(config: HFDataConfig, force_regen: bool = False) -> DatasetDict:
    if os.path.exists(config.splits_dir) and not force_regen:
        logger.info(
            "Full splits already exist at %s, loading from disk.",
            config.splits_dir,
        )
        return config.get_full_splits()

    logger.info("Generating full train and test splits from %s", config.parent_data_dir)
    data = config.get_parent_data()
    data = data.select_columns(["text", "label"])
    data = data.rename_column("label", "subreddit")

    def clean(row):
        row["text"] = row["text"].replace("\n", " ")
        row["len"] = len(row["text"])
        return row

    data = data.map(clean, num_proc=num_proc)
    data = data.shuffle(seed=config.seed)

    if config.task == "one-vs-all":
        # target categories defines the positive class for both the train and test sets
        targets = data.filter(
            lambda x: x["subreddit"] in config.target_categories["train"],
            num_proc=num_proc,
        )
        train_targets, test_targets = hf_train_test_split(
            targets, test_size=0.5, seed=config.seed
        )
        non_targets = data.filter(
            lambda x: x["subreddit"] not in config.target_categories["train"],
            num_proc=num_proc,
        )
    else:
        # target categories defines the positive class for train and test separately
        train_targets = data.filter(
            lambda x: x["subreddit"] in config.target_categories["train"],
            num_proc=num_proc,
        )
        test_targets = data.filter(
            lambda x: x["subreddit"] in config.target_categories["test"],
            num_proc=num_proc,
        )
        non_targets = data.filter(
            lambda x: x["subreddit"] not in config.target_categories["train"]
            and x["subreddit"] not in config.target_categories["test"],
            num_proc=num_proc,
        )

    n_train_targets = len(train_targets)
    n_test_targets = len(test_targets)
    n_total_targets = n_train_targets + n_test_targets
    n_total_non_targets = len(non_targets)

    n_train_non_targets = req_non_targets_from_imbalance(
        n_train_targets, config.train_imbalance, n_total_non_targets, n_total_targets
    )

    # Get required number of non-target samples for train set
    train_non_targets, test_non_targets = hf_train_test_split(
        non_targets, train_size=n_train_non_targets, seed=config.seed
    )
    n_test_non_targets = len(test_non_targets)

    logger.info(
        (
            "Train targets: %d | Train non-targets: %d | "
            "Remaining targets: %d | Remaining non-targets: %d"
        ),
        n_train_targets,
        n_train_non_targets,
        n_test_targets,
        n_test_non_targets,
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
        seed=config.seed
    )

    os.makedirs(config.splits_dir, exist_ok=True)
    full_splits = DatasetDict(
        {
            "train": train_data,
            "test_targets": test_targets,
            "test_non_targets": test_non_targets,
        }
    )
    full_splits.save_to_disk(config.full_splits_dir, num_proc=num_proc)
    logger.info("Saved full splits to %s", config.full_splits_dir)
    return full_splits


def req_non_targets_from_imbalance(
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


def subset_test_split(
    targets: Dataset, non_targets: Dataset, config: HFDataConfig
) -> Dataset:
    # Compute the number of non-target samples for the test sets, either using the
    # requested imbalance ratio or preserving the natural imbalance in the dataset.
    n_targets = len(targets)
    n_total_non_targets = len(non_targets)

    req_non_targets = req_non_targets_from_imbalance(
        n_targets, config.test_imbalance, n_total_non_targets, n_targets
    )

    if req_non_targets > n_total_non_targets:
        # try to subset the test data to get the required level of imbalance
        # don't touch the train data so it can be common across many test set configs
        logger.info(
            "The requested imbalance ratios are too high to use all the target data, "
            "subsetting the test target set to get the required level of imbalance."
        )
        test_imbalance = (
            config.test_imbalance
            if config.test_imbalance is not None
            else n_targets / (n_targets + n_total_non_targets)
        )

        if n_total_non_targets < 100:
            msg = (
                "The requested train imbalance ratio is too high, only "
                "{n_test_non_targets} non-target samples available for test set."
            )
            logger.error(msg)
            raise ValueError(msg)

        n_test_targets = int(test_imbalance * n_total_non_targets)
        if n_test_targets < 10:
            msg = (
                "The requested imbalance ratios are too high, the test set would "
                f"contain only {n_test_targets} target samples."
            )
            logger.error(msg)
            raise ValueError(msg)
        if n_test_targets < 100:
            logger.warning(
                (
                    "The requested imbalance ratios are higher than ideal for this "
                    "dataset, the test set will contain only %d target samples."
                ),
                n_test_targets,
            )

        targets = targets.take(n_test_targets)
        n_targets = len(targets)
        req_non_targets = n_total_non_targets

    logger.info("Test targets: %d | Test non-targets: %d", n_targets, req_non_targets)

    # Get required number of non-target samples
    if req_non_targets > n_total_non_targets:
        msg = "Data generation failed, not enough non-target samples remaining."
        logger.error(msg)
        raise ValueError(msg)
    if req_non_targets != n_total_non_targets:
        _, non_targets = hf_train_test_split(
            non_targets, test_size=req_non_targets, seed=config.seed
        )

    return concatenate_datasets([targets, non_targets]).shuffle(seed=config.seed)


def main(config: HFDataConfig, force_regen: bool):
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
        config.target_categories[split] = [
            x.lower() for x in config.target_categories[split]
        ]

    data_dict = get_full_splits(config, force_regen=force_regen)
    test_data = subset_test_split(
        data_dict["test_targets"], data_dict["test_non_targets"], config
    )

    os.makedirs(config.test_dir, exist_ok=True)
    test_data.save_to_disk(config.test_dir)
    print(f"Saved: {config.test_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument("data_config", type=str, help="Path to the data config file")
    parser.add_argument(
        "--force_regen",
        action="store_true",
        help="Force regeneration of the full splits even if they already exist.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = HFDataConfig.from_path(args.data_config)
    main(config, args.force_regen)
