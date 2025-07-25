import logging
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

from arc_tigers.data.config import HFDataConfig
from arc_tigers.data.utils import hf_train_test_split, tokenize_data
from arc_tigers.sampling.metrics import compute_metrics, evaluate
from arc_tigers.training.config import TrainConfig
from arc_tigers.utils import seed_everything, to_json

logger = logging.getLogger(__name__)


def train_transformers(train_config: TrainConfig):
    if not isinstance(train_config.data_config, HFDataConfig):
        msg = "Training a classifier on synthetic data is not supported."
        raise ValueError(msg)
    if train_config.model_config.is_synthetic:
        msg = "Training a classifier with a synthetic model is not supported."
        raise ValueError(msg)
    if train_config.hparams_config is None:
        msg = "Training configuration must include a hyperparameters configuration."
        raise ValueError(msg)

    save_dir = train_config.model_dir
    os.makedirs(save_dir, exist_ok=False)

    logging.basicConfig(filename=f"{save_dir}/logs.log", level=logging.INFO)

    train_config.save(save_dir)
    logger.info("Experiment configuration saved to %s", save_dir)
    seed = train_config.hparams_config.train_kwargs["seed"]
    seed_everything(seed)

    # Load tokenizer and model
    logger.info("Loading model and tokenizer...")
    model_name = train_config.model_config.model_id
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name, **train_config.model_config.model_kwargs
    )

    # Data collator
    logger.info("Loading data...")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = train_config.data_config.get_train_split()
    logger.info("Tokenizing training data...")
    train_dataset = tokenize_data(train_dataset, tokenizer)
    logger.info("Creating val split...")
    train_dataset, eval_dataset = hf_train_test_split(
        train_dataset, seed=seed, test_size=0.1
    )

    logger.info("Starting training...")
    training_args = TrainingArguments(
        output_dir=str(save_dir),
        logging_dir=f"{save_dir}/logs",
        **train_config.hparams_config.train_kwargs,
    )
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    logger.info("Running final evaluation...")
    results = trainer.evaluate()
    logger.info("Evaluation results: %s", results)
    # Save evaluation results to a JSON file
    results_path = save_dir / "evaluation_results.json"
    to_json(results, results_path)
    logger.info("Evaluation results saved to %s", results_path)

    # Save the model and tokenizer
    logger.info("Saving model and tokenizer to %s", save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


def train_tfidf(train_config: TrainConfig):
    if not isinstance(train_config.data_config, HFDataConfig):
        msg = "Training a classifier on synthetic data is not supported."
        raise ValueError(msg)
    if train_config.model_config.is_synthetic:
        msg = "Training a classifier with a synthetic model is not supported."
        raise ValueError(msg)
    if train_config.hparams_config is None:
        msg = "Training configuration must include a hyperparameters configuration."
        raise ValueError(msg)

    save_dir = train_config.model_dir
    os.makedirs(save_dir, exist_ok=False)

    logging.basicConfig(filename=f"{save_dir}/logs.log", level=logging.INFO)

    train_config.save(save_dir)
    logger.info("Experiment configuration saved to %s", save_dir)
    seed = train_config.hparams_config.train_kwargs.pop("seed")
    seed_everything(seed)

    # Load model
    pipeline = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(
            **train_config.hparams_config.train_kwargs, random_state=seed
        ),
    )

    train_dataset = train_config.data_config.get_train_split()
    train_dataset, eval_dataset = hf_train_test_split(
        train_dataset, seed=seed, test_size=0.1
    )

    # Train the model
    pipeline.fit(train_dataset["text"], train_dataset["label"])

    # Evaluate the model
    eval_predictions = pipeline.predict_proba(eval_dataset["text"])
    eval_results = evaluate(eval_dataset, eval_predictions)
    print("Evaluation results:", eval_results)
    # Save evaluation results to a JSON file
    results_path = save_dir / "evaluation_results.json"
    to_json(eval_results, results_path)

    logger.info("Evaluation results saved to %s", results_path)

    # Save the model and tokenizer
    joblib.dump(pipeline, save_dir / f"{train_config.model_config.model_id}.joblib")
