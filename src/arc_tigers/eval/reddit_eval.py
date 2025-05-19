import json
import os

import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from arc_tigers.data.reddit_data import get_reddit_data
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.utils import load_yaml


def get_preds(
    data_config_path: str,
    save_dir: str,
    class_balance: float,
    seed: int,
) -> tuple[np.ndarray, Dataset]:
    data_config = load_yaml(data_config_path)

    # calculate predictions for whole dataset
    print(f"Loading model and tokenizer from {save_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(save_dir)
    model = AutoModelForSequenceClassification.from_pretrained(save_dir)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    if class_balance != 1.0:
        data_config["data_args"]["balanced"] = False

    _, _, test_dataset, meta_data = get_reddit_data(
        **data_config["data_args"],
        tokenizer=tokenizer,
        random_seed=seed,
        class_balance=class_balance,
    )
    # Save meta_data to the save_dir
    meta_data_path = os.path.join(save_dir, "data_stats.json")
    with open(meta_data_path, "w") as meta_file:
        json.dump(meta_data, meta_file, indent=2)

    training_args = TrainingArguments(output_dir="tmp", per_device_eval_batch_size=16)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    preds = trainer.predict(test_dataset, metric_key_prefix="").predictions
    return preds, test_dataset
