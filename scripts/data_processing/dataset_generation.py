import glob
import json
import os

import pandas as pd
from tqdm import tqdm

from arc_tigers.data.reddit_data import DATA_DRIFT_COMBINATIONS, ONE_VS_ALL_COMBINATIONS
from arc_tigers.data.utils import clean_row, flag_row, is_valid_row


def process_data(data, target_categories, args, save_path, shard_id=None):
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

    train_data_targets = []
    test_data_targets = []
    non_targets = [[], []]

    for row_index, row in enumerate(tqdm(data)):
        if flag_row(row):
            continue
        if args.mode == "one-vs-all":
            # Split target classes evenly between train and test
            if row["communityName"] in target_categories["train"]:
                # Alternate assignment for even split
                if row_index % 2 == 0:
                    train_data_targets.append(clean_row(row))
                else:
                    test_data_targets.append(clean_row(row))
            else:
                non_targets[row_index % 2].append(clean_row(row))
        else:
            if row["communityName"] in target_categories["train"]:
                train_data_targets.append(clean_row(row))
            elif row["communityName"] in target_categories["test"]:
                test_data_targets.append(clean_row(row))
            else:
                non_targets[row_index % 2].append(clean_row(row))

    # Imbalance the dataset
    if args.r is not None:
        imbalance_ratio = float(args.r)
        if imbalance_ratio > 1:
            n_targets = int(imbalance_ratio)
        else:
            n_targets = int(len(non_targets[0]) * imbalance_ratio)
        n_train_targets = n_targets
        n_test_targets = n_targets
    else:
        n_train_targets = len(train_data_targets)
        n_test_targets = len(test_data_targets)

    # Filter out invalid rows
    train_targets_clean = [
        r for r in train_data_targets[:n_train_targets] if is_valid_row(r)
    ]
    test_targets_clean = [
        r for r in test_data_targets[:n_test_targets] if is_valid_row(r)
    ]
    train_non_targets_clean = [r for r in non_targets[0] if is_valid_row(r)]
    test_non_targets_clean = [r for r in non_targets[1] if is_valid_row(r)]

    # sorr the data by length to get the most informative samples first
    train_targets_df = pd.DataFrame.from_dict(train_targets_clean).sort_values(
        "len", ascending=False, inplace=False
    )
    test_targets_df = pd.DataFrame.from_dict(test_targets_clean).sort_values(
        "len", ascending=False, inplace=False
    )
    train_non_targets_df = pd.DataFrame.from_dict(train_non_targets_clean).sort_values(
        "len", ascending=False, inplace=False
    )
    test_non_targets_df = pd.DataFrame.from_dict(test_non_targets_clean).sort_values(
        "len", ascending=False, inplace=False
    )

    # concatenate targets with non-targets
    train_data = pd.concat([train_targets_df, train_non_targets_df])
    test_data = pd.concat([test_targets_df, test_non_targets_df])

    # if using sharding save data to corresponding shard id
    if shard_id is not None:
        train_csv = f"{save_path}/train_shard_{shard_id}.csv"
        test_csv = f"{save_path}/test_shard_{shard_id}.csv"
    # otherwise save to train.csv and test.csv
    else:
        train_csv = f"{save_path}/train.csv"
        test_csv = f"{save_path}/test.csv"

    train_data.to_csv(train_csv, index=False)
    test_data.to_csv(test_csv, index=False)
    print(
        f"Shard {shard_id if shard_id is not None else ''}: "
        f"Train data: {len(train_data)} | Test data: {len(test_data)}"
    )
    print(f"Saved: {train_csv}, {test_csv}")


def main(args):
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

    # If the data_dir is a directory, process each shard
    if os.path.isdir(args.data_dir):
        shard_files = sorted(
            glob.glob(os.path.join(args.data_dir, "filtered_rows_shard_*.json"))
        )
        for i, shard_file in enumerate(shard_files):
            with open(shard_file) as f:
                data = json.load(f)
            process_data(data, target_categories, args, save_path, shard_id=i)
    # If the data_dir is a file, process the single JSON file
    elif args.data_dir.endswith(".json"):
        with open(args.data_dir) as f:
            data = json.load(f)
        process_data(data, target_categories, args, save_path)
    else:
        err_msg = f"Error: '{args.data_dir}' is not a directory or a JSON file."
        raise ValueError(err_msg)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument(
        "data_dir",
        type=str,
        default="data/reddit_dataset_12/10000000_rows/",
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
        help="Splitting mode: 'data-drift' or 'one-vs-all'.",
    )
    parser.add_argument(
        "-r",
        default=None,
        help="Imbalance ratio for the dataset. provide a float or int, "
        "where 0.01 means 1% of the dataset is target classes. Int values are also "
        "accepted, where 100 means 100 samples are the target classes.",
    )
    args = parser.parse_args()
    main(args)
