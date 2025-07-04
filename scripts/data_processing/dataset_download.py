import argparse
import json

from datasets import Dataset, IterableDataset, load_dataset

from arc_tigers.constants import DATA_DIR


def main(
    dataset_name: str, max_rows: int, target_subreddits: str, seed: int, min_length: int
):
    # Load the dataset in streaming mode to avoid downloading the entire dataset
    iter_ds: IterableDataset = load_dataset(dataset_name, streaming=True)["train"]

    # Load the target subreddits
    with open(target_subreddits) as file:
        placeholder_subreddits = json.load(file)
        selected_subreddits = [
            placeholder_subreddit.lower()
            for placeholder_subreddit in placeholder_subreddits
        ]

    # Only keep requested subreddits and filter by minimum length
    iter_ds = iter_ds.filter(
        lambda x: len(x["text"]) >= min_length and x["label"] in selected_subreddits
    )

    # Get max_rows rows from the dataset, shuffling first
    iter_ds = iter_ds.shuffle(seed=seed)
    iter_ds = iter_ds.take(max_rows)
    ds = Dataset.from_generator(lambda: (yield from iter_ds), features=iter_ds.features)

    # Save the dataset
    if target_subreddits.endswith("top_subreddits.json"):
        save_path = f"{DATA_DIR}/{dataset_name.split('/')[-1]}"
    else:
        save_path = f"{DATA_DIR}/{target_subreddits.split('/')[-1].rstrip('.json')}"

    ds.save_to_disk(save_path)


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

    main(
        args.dataset_name,
        args.max_rows,
        args.target_subreddits,
        args.seed,
        args.min_length,
    )
