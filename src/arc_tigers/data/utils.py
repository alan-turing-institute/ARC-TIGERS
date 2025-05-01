ONE_VS_ALL_COMBINATIONS = {
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

BINARY_COMBINATIONS = {
    "sport": {
        "train": ["r/soccer", "r/Cricket"],
        "test": ["r/soccer", "r/Cricket"],
    },
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
