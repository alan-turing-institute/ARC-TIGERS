from collections import Counter

from datasets import concatenate_datasets


def flag_row(row: dict) -> bool:
    """
    Flags a row for removal if it is empty, deleted, or removed. It also flags
    rows that are too short (less than 10 characters). It also flags rows
    that are posts and not comments.

    Args:
        row: row from the reddit dataset

    Returns:
        bool True if the row should be flagged for removal, False otherwise
    """
    if not row["text"]:
        return True
    if row["text"] == "[deleted]":
        return True
    if row["text"] == "[removed]":
        return True
    if len(row["text"]) < 10:
        return True
    return row["dataType"] == "post"


def balance_dataset(dataset):
    # Get the number of samples in each class
    label_counts = Counter(dataset["label"])
    # Calculate the number for each class
    min_samples = min(label_counts.values())
    resampled_data = []
    for label in label_counts:
        label_data = dataset.filter(lambda x, label=label: x["label"] == label)
        resampled_data.append(label_data.shuffle().select(range(min_samples)))
    return concatenate_datasets(resampled_data)


def clean_row(row: dict) -> dict:
    """
    The function takse a row of the reddit dataset with all of the fields. It then
    removes the extra fields and returns a new row with only the relevant fields
    (text, label, and the length). It also renames these appropriately.
    Args:
        row: row from the reddit dataset

    Returns:
        new_row: a new row with only the relevant fields
    """
    new_row = {}
    new_row["text"] = row["text"]
    new_row["label"] = row["communityName"]
    new_row["len"] = len(row["text"])
    return new_row


def get_target_mapping(eval_setting, target_subreddits):
    if eval_setting == "multi-class":
        return {subreddit: index for index, subreddit in enumerate(target_subreddits)}
    if eval_setting == "one-vs-all":
        return dict.fromkeys(target_subreddits, 1)
    err_msg = f"Invalid evaluation setting: {eval_setting}"
    raise ValueError(err_msg)


def preprocess_function(examples, tokenizer, targets: dict[str, int]):
    tokenized = tokenizer(examples["text"], padding=True, truncation=True)
    tokenized["label"] = [targets.get(label, 0) for label in examples["label"]]
    return tokenized
