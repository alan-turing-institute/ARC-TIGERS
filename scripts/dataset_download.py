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
    # Ensure the output directory exists
    os.makedirs("data/", exist_ok=True)

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
    data = []
    for i, example in enumerate(tqdm(ds, desc="Loading dataset", total=max_rows)):
        data.append(example)
        if i >= max_rows:
            break

    # Save the filtered data to a JSON file
    if target_subreddits == "data/top_subreddits.json":
        dataset_name = dataset_name.split("/")[-1]
    else:
        dataset_name = target_subreddits.split("/")[-1].rstrip(".json")
    output_dir = f"{DATA_DIR}/{dataset_name}/{max_rows}_rows/"
    os.makedirs(output_dir, exist_ok=True)
    save_pth = f"{output_dir}/filtered_rows.json"
    json_data = json.dumps(data, indent=2)
    with open(save_pth, "w") as f:
        f.write(json_data)
    print(f"Data saved to {save_pth}")


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
    args = parser.parse_args()

    main(args)
