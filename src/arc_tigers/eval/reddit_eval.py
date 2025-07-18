import json
import os
from glob import glob

import joblib
import numpy as np
from datasets import Dataset
from sklearn.pipeline import Pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from arc_tigers.constants import DATA_CONFIG_DIR, MODEL_CONFIG_DIR
from arc_tigers.data.reddit_data import get_reddit_data
from arc_tigers.data.synthetic import get_synthetic_data
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.model.beta_model import BetaModel
from arc_tigers.utils import load_yaml


def get_train_data_from_exp_dir(exp_dir: str) -> Dataset:
    # get the training dataset for the surrogate model
    model_exp_config = load_yaml(f"{exp_dir}/experiment_config.json")
    surrogate_model_config = load_yaml(
        f"{MODEL_CONFIG_DIR}/{model_exp_config['model_config']}.yaml"
    )
    surrogate_data_config = load_yaml(
        f"{DATA_CONFIG_DIR}/{model_exp_config['data_config']}.yaml"
    )
    tokenizer = AutoTokenizer.from_pretrained(surrogate_model_config["model_id"])

    surrogate_training_dataset, _, _, _ = get_reddit_data(
        **surrogate_data_config["data_args"],
        tokenizer=tokenizer,
    )
    return surrogate_training_dataset


def get_preds(*args, **kwargs):
    """
    Wrapper function to get predictions from either a transformer model or a tfidf
    model.
    """
    if len(glob(f"{kwargs['save_dir']}/*.joblib")) > 0:
        return get_tfidf_preds(*args, **kwargs)
    return get_transformers_preds(*args, **kwargs)


def get_transformers_preds(
    data_config_path: str,
    model_config_path: str,
    save_dir: str,
    class_balance: float,
    seed: int,
    preds_exist: bool = False,
) -> tuple[np.ndarray | None, Dataset]:
    """
    Get the predictions from a model using the transformers library. This function is
    also used when synthetic data is used. Model weights are loaded from the save_dir
    data and model config data are loaded from the data_config_path and
    model_config_path. The seed should be set to the same value as used in the training.

    Args:
        data_config_path: Data config path
        model_config_path: Model config path
        save_dir: Model weights directory
        class_balance: The class balance to use in evaluation.
        seed: The seed to use for the random number generator, this should be the same
            as the seed used in training.

    Returns:
        tuple `preds` the predictions from the model on the test dataset. `test_dataset`
        the test dataset used for predictions.
    """
    model_config = load_yaml(model_config_path)
    if model_config["model_id"] == "beta_model":
        # Arbitrary tokenizer only loaded for compatibility with other functions, not
        # used by the synthetic Beta model.
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        use_cpu = True
    else:
        # calculate predictions for whole dataset
        print(f"Loading model and tokenizer from {save_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(model_config["model_id"])
        model = AutoModelForSequenceClassification.from_pretrained(save_dir)
        use_cpu = False

    data_config = load_yaml(data_config_path)

    if data_config["data_name"] == "synthetic":
        n_samples = data_config["data_args"]["n_rows"]
        negative_samples = int(n_samples / (1 + class_balance))
        positive_samples = n_samples - negative_samples
        imbalance = positive_samples / n_samples
        test_dataset = get_synthetic_data(
            n_samples, imbalance, seed=seed, tokenizer=tokenizer
        )
        meta_data = {
            "train_label_counts": {
                "n_class_0": int((np.array(test_dataset["label"]) == 0).sum()),
                "n_class_1": int((np.array(test_dataset["label"]) == 1).sum()),
            }
        }
    else:
        if class_balance != 1.0:
            data_config["data_args"]["balanced"] = False

        if data_config["data_args"].get("class_balance") is not None:
            err_msg = (
                f"Selected data config '{data_config_path}' already "
                "has a class imbalance, please specify balanced data config."
            )
            raise ValueError(err_msg)

        _, _, test_dataset, meta_data = get_reddit_data(
            **data_config["data_args"],
            tokenizer=tokenizer,
            class_balance=class_balance,
        )

    if preds_exist:
        return None, test_dataset

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Save meta_data to the save_dir
    meta_data_path = os.path.join(save_dir, "data_stats.json")
    with open(meta_data_path, "w") as meta_file:
        json.dump(meta_data, meta_file, indent=2)

    if model_config["model_id"] == "beta_model":
        n_class_0 = (np.array(test_dataset["label"]) == 0).sum()
        n_class_1 = (np.array(test_dataset["label"]) == 1).sum()
        # BetaModel uses a different definition of imbalance than the class_balance
        # used in the rest of this script - we should settle on one definition.
        # The BetaModel form is the proportion of data in the minority class
        imbalance = n_class_1 / (n_class_0 + n_class_1)
        model_kwargs = model_config["model_kwargs"]
        pos_err_rate = model_kwargs["positive_error_rate"]
        neg_err_rate = model_kwargs["negative_error_rate"]
        if pos_err_rate or neg_err_rate:
            if not (pos_err_rate and neg_err_rate):
                msg = "Both positive and negative error rates must be set."
                raise ValueError(msg)
            model = BetaModel.from_error_rates(pos_err_rate, neg_err_rate)
        else:
            model = BetaModel.from_imbalance_and_advantage(
                imbalance, model_kwargs["model_adv"]
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


def get_tfidf_preds(
    data_config_path: str,
    save_dir: str,
    class_balance: float,
    **kwargs,
) -> tuple[np.ndarray, Dataset]:
    """
    Function to get the predictions from a tfidf model. The model is loaded from the
    save_dir and the data is loaded from the data_config_path. The seed should be set
    to the same value as used in the training.

    Args:
        data_config_path: data config path
        save_dir: directory where the model is saved
        class_balance: The class balance to use in evaluation.
        seed: The seed to use for the random number generator, this should be the same
            as the seed used in training.

    Returns:
        tuple `preds` the predictions from the model on the test dataset. `test_dataset`
        the test dataset used for predictions.
    """
    data_config = load_yaml(data_config_path)

    print(f"Loading model and tokenizer from {save_dir} ...")

    tokenizer = None
    # Load the model from the joblib file
    model_path = glob(f"{save_dir}/*.joblib")[0]
    model: Pipeline = joblib.load(model_path)

    # Data collator
    if class_balance != 1.0:
        data_config["data_args"]["balanced"] = False

    _, _, test_dataset, meta_data = get_reddit_data(
        **data_config["data_args"],
        tokenizer=tokenizer,
        class_balance=class_balance,
    )
    # Save meta_data to the save_dir
    meta_data_path = os.path.join(save_dir, "data_stats.json")
    with open(meta_data_path, "w") as meta_file:
        json.dump(meta_data, meta_file, indent=2)

    preds = model.predict(test_dataset["text"])

    # get the training dataset for the surrogate model
    model_exp_config = load_yaml(f"{save_dir}/experiment_config.json")
    surrogate_data_config = load_yaml(model_exp_config["data_config_path"])

    surrogate_training_dataset, _, _, _ = get_reddit_data(
        **surrogate_data_config["data_args"],
        tokenizer=tokenizer,
    )

    data = {
        "evaluation_dataset": test_dataset,
        "surrogate_train_dataset": surrogate_training_dataset,
    }

    return preds, data
