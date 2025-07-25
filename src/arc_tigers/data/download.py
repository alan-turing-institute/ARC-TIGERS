import json
import logging

from datasets import Dataset, IterableDataset, load_dataset

from arc_tigers.constants import DATA_DIR

logger = logging.getLogger(__name__)


def download_reddit(
    dataset_name: str, max_rows: int, target_subreddits: str, seed: int, min_length: int
):
    """
    Downloads a Reddit dataset, filters it by specified subreddits and minimum length,
    and saves it to disk.

    Args:
        dataset_name: HuggingFace name of reddit dataset download.
        max_rows: The maximum number of rows to keep in the dataset.
        target_subreddits: Path to a JSON file containing the subreddits to keep.
        seed: Random seed for shuffling the dataset.
        min_length: Minimum number of characters of text entries to keep in the dataset.
    """
    # Load the dataset in streaming mode to avoid downloading the entire dataset
    iter_ds: IterableDataset = load_dataset(dataset_name, streaming=True)["train"]

    # Load the target subreddits
    with open(target_subreddits) as file:
        placeholder_subreddits = json.load(file)
        selected_subreddits = [
            placeholder_subreddit.lower()
            for placeholder_subreddit in placeholder_subreddits
        ]

    # Only keep requested subreddits, filter by minimum length, and remove posts with
    # "/r/" in the text which may reveal the subreddit label and make the task too easy
    iter_ds = iter_ds.filter(
        lambda x: len(x["text"]) >= min_length
        and x["label"] in selected_subreddits
        and "/r/" not in x["text"]
    )

    # Get max_rows rows from the dataset, shuffling first
    iter_ds = iter_ds.shuffle(seed=seed)
    iter_ds = iter_ds.take(max_rows)
    ds = Dataset.from_generator(lambda: (yield from iter_ds), features=iter_ds.features)

    # Save the dataset
    if target_subreddits.endswith("top_subreddits.json"):
        save_path = DATA_DIR / f"{dataset_name.split('/')[-1]}"
    else:
        save_path = DATA_DIR / f"{target_subreddits.split('/')[-1].rstrip('.json')}"

    ds.save_to_disk(save_path)
    logger.info("Saved dataset to %s", save_path)
