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


def flag_row(row):
    if not row["text"]:
        return True
    if row["text"] == "[deleted]":
        return True
    if row["text"] == "[removed]":
        return True
    if len(row["text"]) < 10:
        return True
    return row["dataType"] == "post"


def clean_row(row):
    new_row = {}
    new_row["text"] = row["text"]
    new_row["label"] = row["communityName"]
    new_row["len"] = len(row["text"])
    return new_row
