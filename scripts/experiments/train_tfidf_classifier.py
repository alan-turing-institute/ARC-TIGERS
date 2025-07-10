import argparse
import logging
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from arc_tigers.data.config import HFDataConfig
from arc_tigers.data.utils import hf_train_test_split
from arc_tigers.sampling.metrics import evaluate
from arc_tigers.training.config import TrainConfig
from arc_tigers.utils import seed_everything, to_json

logger = logging.getLogger(__name__)


def main(args):
    train_config = TrainConfig.from_path(args.train_config)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a TF-IDF classifier")
    parser.add_argument("train_config", help="path to the train config yaml file")
    args = parser.parse_args()
    main(args)
