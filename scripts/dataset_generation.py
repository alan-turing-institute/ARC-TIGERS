import json
import os

import pandas as pd
from tqdm import tqdm

from arc_tigers.data.utils import DATASET_COMBINATIONS, clean_row, flag_row


def main(args):
    """
    Takes as an argument a json file containing reddit comments and posts, and generates
      a dataset which uses the associated subreddit as a label. The dataset comprsises
      of a train set, and a test set, which have related but different subreddits for
      the purposes of evaluation. The target class in a given dataset is poorly
      represented, with the ratio of target classes to non target classes being defined
      by the imbalance ratio

    Args:
        args: _description_
    """

    target_categories = DATASET_COMBINATIONS[args.target_config]
    save_dir = "/".join(args.data_dir.split("/")[:-1])

    with open(args.data_dir) as f:
        data = json.load(f)

    train_data_targets = []
    test_data_targets = []

    non_targets = [[], []]

    for row_index, row in enumerate(tqdm(data)):
        # flag the row if it is empty, deleted, removed, or too short
        # and skip it
        if flag_row(row):
            continue
        if row["communityName"] in target_categories["train"]:
            train_data_targets.append(clean_row(row))
        elif row["communityName"] in target_categories["test"]:
            test_data_targets.append(clean_row(row))
        else:
            non_targets[row_index % 2].append(clean_row(row))

    # Imbalance the dataset
    if args.r is not None:
        imbalance_ratio = args.r
        if isinstance(imbalance_ratio, int):
            n_targets = imbalance_ratio
        else:
            n_targets = int(len(non_targets[0]) * imbalance_ratio)

        n_train_targets = n_targets
        n_test_targets = n_targets
    else:
        # None will just use all values in array
        n_train_targets = len(train_data_targets)
        n_test_targets = len(test_data_targets)

    train_data = train_data_targets[:n_targets] + non_targets

    train_targets_df = pd.DataFrame.from_dict(
        train_data_targets[:n_train_targets]
    ).sort_values("len", ascending=False, inplace=False)

    test_targets_df = pd.DataFrame.from_dict(
        test_data_targets[:n_test_targets]
    ).sort_values("len", ascending=False, inplace=False)

    train_non_targets_df = pd.DataFrame.from_dict(non_targets[0]).sort_values(
        "len", ascending=False, inplace=False
    )
    test_non_targets_df = pd.DataFrame.from_dict(non_targets[1]).sort_values(
        "len", ascending=False, inplace=False
    )

    train_data = pd.concat([train_targets_df, train_non_targets_df])
    test_data = pd.concat([test_targets_df, test_non_targets_df])

    save_path = f"{save_dir}/splits/{args.split}/"
    os.makedirs(save_path, exist_ok=True)
    train_data.to_csv(f"{save_path}/train.csv", index=False)
    test_data.to_csv(f"{save_path}/test.csv", index=False)
    print(f"Train data: {len(train_data)} total | {n_train_targets} targets")
    print(f"Test data: {len(test_data)} total | {n_test_targets} targets")
    print(f"Train data saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument(
        "data_dir",
        type=str,
        default="../data/reddit_dataset_12/15000000_rows/filtered_rows.json",
        help="Path to the data used for generation",
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
        "-r",
        default=None,
        help="Imbalance ratio for the dataset. provide a float or int, "
        "where 0.01 means 1% of the dataset is target classes. Int values are also "
        "accepted, where 100 means 100 samples are the target classes.",
    )
    args = parser.parse_args()
    main(args)
