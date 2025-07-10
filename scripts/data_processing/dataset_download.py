import argparse

from arc_tigers.constants import DATA_DIR
from arc_tigers.data.download import download_reddit

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and filter a dataset.")
    parser.add_argument(
        "--max_rows", type=int, required=True, help="Maximum number of rows to process."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bit0/reddit_dataset_12",
        help="Name of the dataset to load.",
    )
    parser.add_argument(
        "--target_subreddits",
        type=str,
        default=f"{DATA_DIR}/top_subreddits.json",
        help="Optional: Path to a JSON file containing a list of target subreddits.",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=30,
        help="Optional: Minimum length of posts/comments to keep, in characters.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional: Random seed for shuffling.",
    )
    args = parser.parse_args()

    download_reddit(
        args.dataset_name,
        args.max_rows,
        args.target_subreddits,
        args.seed,
        args.min_length,
    )
