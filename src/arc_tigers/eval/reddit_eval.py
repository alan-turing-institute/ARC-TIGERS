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
from arc_tigers.data.synthetic import get_synthetic_data
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.model.beta_model import BetaModel
from arc_tigers.utils import load_yaml


def get_preds(
    data_config_path: str,
    model_config_path: str,
    save_dir: str,
    class_balance: float,
    seed: int,
    synthetic_args: dict,
) -> tuple[np.ndarray, Dataset]:
    if model_config_path == "beta_model":
        # Arbitrary tokenizer only loaded for compatibility with other functions, not
        # used by the ssynthetic Beta model.
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        use_cpu = True
    else:
        # calculate predictions for whole dataset
        print(f"Loading model and tokenizer from {save_dir}...")
        model_config = load_yaml(model_config_path)
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        use_cpu = False

    if data_config_path == "synthetic":
        negative_samples = int(
            synthetic_args["synthetic_samples"] / (1 + class_balance)
        )
        positive_samples = synthetic_args["synthetic_samples"] - negative_samples
        imbalance = positive_samples / synthetic_args["synthetic_samples"]
        test_dataset = get_synthetic_data(
            synthetic_args["synthetic_samples"],
            imbalance,
            seed=seed,
            tokenizer=tokenizer,
        )
        meta_data = {
            "train_label_counts": {
                "n_class_0": int((np.array(test_dataset["label"]) == 0).sum()),
                "n_class_1": int((np.array(test_dataset["label"]) == 1).sum()),
            }
        }
    else:
        data_config = load_yaml(data_config_path)
        if class_balance != 1.0:
            data_config["data_args"]["balanced"] = False

        _, _, test_dataset, meta_data = get_reddit_data(
            **data_config["data_args"],
            tokenizer=tokenizer,
            random_seed=seed,
            class_balance=class_balance,
        )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Save meta_data to the save_dir
    meta_data_path = os.path.join(save_dir, "data_stats.json")
    with open(meta_data_path, "w") as meta_file:
        json.dump(meta_data, meta_file, indent=2)

    if model_config_path == "beta_model":
        n_class_0 = (np.array(test_dataset["label"]) == 0).sum()
        n_class_1 = (np.array(test_dataset["label"]) == 1).sum()
        # BetaModel uses a different definition of imbalance than the class_balance
        # used in the rest of this script - we should settle on one definition.
        # The BetaModel form is the proportion of data in the minority class
        imbalance = n_class_1 / (n_class_0 + n_class_1)
        if (
            synthetic_args["positive_error_rate"]
            or synthetic_args["negative_error_rate"]
        ):
            if not (
                synthetic_args["positive_error_rate"]
                and synthetic_args["negative_error_rate"]
            ):
                msg = "Both positive and negative error rates must be set."
                raise ValueError(msg)
            model = BetaModel.from_error_rates(
                synthetic_args["positive_error_rate"],
                synthetic_args["negative_error_rate"],
            )
        else:
            model = BetaModel.from_imbalance_and_advantage(
                imbalance, synthetic_args["model_adv"]
            )

    training_args = TrainingArguments(
        output_dir="tmp",
        per_device_eval_batch_size=16,
        use_cpu=use_cpu,
    )
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
