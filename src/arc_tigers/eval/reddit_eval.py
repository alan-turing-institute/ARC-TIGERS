import logging
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
from arc_tigers.data.config import HFDataConfig, SyntheticDataConfig
from arc_tigers.data.reddit_data import get_reddit_data
from arc_tigers.data.utils import tokenize_data
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.model.beta_model import BetaModel
from arc_tigers.sample.utils import get_preds_cache_path
from arc_tigers.training.config import TrainConfig
from arc_tigers.utils import load_yaml

logger = logging.getLogger(__name__)


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


def get_preds(
    train_config: TrainConfig,
    data_config: HFDataConfig | SyntheticDataConfig,
    test_dataset: Dataset | None = None,
) -> np.ndarray:
    """
    Wrapper function to get predictions from either a transformer model or a tfidf
    model.
    """
    cache_path = get_preds_cache_path(train_config, "test", data_config)
    preds_exist = os.path.exists(cache_path)
    if preds_exist:
        logger.info("Loading cached predictions from %s", cache_path)
        return np.load(cache_path)

    if test_dataset is None:
        test_dataset = data_config.get_test_split()

    if len(glob(f"{train_config.save_dir}/*.joblib")) > 0:
        preds = get_tfidf_preds(train_config, test_dataset)
    preds = get_transformers_preds(
        train_config, test_dataset, data_config.test_imbalance
    )

    os.makedirs(cache_path.parent, exist_ok=True)
    np.save(cache_path, preds)
    logger.info("Saved predictions to %s", cache_path)

    return preds


def get_transformers_preds(
    train_config: TrainConfig, test_dataset: Dataset, imbalance: float | None = None
) -> np.ndarray:
    """
    Get the predictions from a model using the transformers library. This function is
    also used when synthetic data is used.

    Args:
        train_config: Train config, specifies the previously trained model to use
        data_config: Data config, specifies the test dataset to use for predictions.
        test_dataset: Optional pre-loaded test dataset from data config.
        imbalance: Optional imbalance ratio for if train_config specifies a synthetic
            model

    Returns:
        predictions from the model on the test dataset
    """
    model_path = train_config.save_dir
    if train_config.model_config.is_synthetic:
        # Arbitrary tokenizer only loaded for compatibility with other functions, not
        # used by the synthetic Beta model.
        logger.info("Loading roberta-base tokenizer as placeholder for synthetic model")
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        use_cpu = True
    else:
        # calculate predictions for whole dataset
        logger.info("Loading tokenizer from %s...", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        use_cpu = False

    test_dataset = tokenize_data(test_dataset, tokenizer)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if train_config.model_config.is_synthetic:
        if imbalance is None:
            msg = "Test imbalance must be set for synthetic models."
            raise ValueError(msg)

        model_kwargs = train_config.model_config.model_kwargs
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
    else:
        logger.info("Loading model from %s...", model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

    training_args = TrainingArguments(
        output_dir="tmp", per_device_eval_batch_size=16, use_cpu=use_cpu
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Computing predictions...")
    return trainer.predict(test_dataset, metric_key_prefix="").predictions


def get_tfidf_preds(train_config: TrainConfig, test_dataset: Dataset) -> np.ndarray:
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
    logger.info("Loading model from %s...", train_config.save_dir)
    model_path = glob(f"{train_config.save_dir}/*.joblib")[0]
    model: Pipeline = joblib.load(model_path)

    return model.predict(test_dataset["text"])
