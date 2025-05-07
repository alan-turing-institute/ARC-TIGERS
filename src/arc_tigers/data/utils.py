DATASET_COMBINATIONS = {
    "sport": {
        "train": ["r/soccer", "r/Cricket"],
        "test": ["r/nfl", "NFLv2", "r/NBATalk", "r/nba"],
    },
    "american_football": {
        "train": ["r/nfl", "NFLv2"],
        "test": ["r/fantasyfootball"],
    },
    "ami": {"train": ["r/AmItheAsshole"], "test": ["r/AmIOverreacting"]},
    "news": {"train": ["r/news"], "test": ["r/Worldnews"]},
    "advice": {"train": ["r/AskReddit"], "test": ["r/AskMenAdvice", "r/Advice"]},
}


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
