import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from arc_tigers.constants import DATA_DIR


def main(args):
    dataset_name = args.dataset_name
    max_rows = args.max_rows
    target_subreddits = args.target_subreddits
    shard_size = args.shard_size

    # Load the dataset in streaming mode to avoid downloading the entire dataset
    unfiltered = load_dataset(dataset_name, streaming=True)["train"]

    # Load the target subreddits
    with open(target_subreddits) as file:
        placeholder_subreddits = json.load(file)
        selected_subreddits = [
            placeholder_subreddit.lower()
            for placeholder_subreddit in placeholder_subreddits
        ]

    # Filter the dataset
    ds = unfiltered.filter(
        lambda example: example["communityName"].lower() in selected_subreddits
    )
    # Process the dataset
    if target_subreddits == "data/top_subreddits.json":
        dataset_name_out = dataset_name.split("/")[-1]
    else:
        dataset_name_out = target_subreddits.split("/")[-1].rstrip(".json")

    # Create output directory
    output_dir = f"{DATA_DIR}/{dataset_name_out}/{max_rows}_rows/"
    os.makedirs(output_dir, exist_ok=True)

    # Save the filtered data to JSON files

    data = []
    # If shard_size is None or 0, process without sharding
    if not shard_size:
        for i, example in enumerate(tqdm(ds, desc="Loading dataset", total=max_rows)):
            data.append(example)
            if i + 1 >= max_rows:
                break
        save_pth = f"{output_dir}/filtered_rows.json"
        with open(save_pth, "w") as f:
            json.dump(data, f, indent=2)
        print(f"All data saved to {save_pth}")
        return

    # Save the filtered data to JSON files
    else:
        shard_idx = 0
        for i, example in enumerate(tqdm(ds, desc="Loading dataset", total=max_rows)):
            data.append(example)
            if (i + 1) % shard_size == 0 or i + 1 == max_rows:
                save_pth = f"{output_dir}/filtered_rows_shard_{shard_idx}.json"
                with open(save_pth, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"Shard {shard_idx} saved to {save_pth}")
                data = []
                shard_idx += 1
            if i + 1 >= max_rows:
                break
    return


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
        default="data/top_subreddits.json",
        help="Optional: Path to a JSON file containing a list of target subreddits.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=None,
        help="Optional: Number of rows per shard.",
    )
    args = parser.parse_args()

    main(args)
