import argparse
import json
import logging
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from arc_tigers.data.reddit_data import get_reddit_data
from arc_tigers.eval.utils import tfidf_evaluation
from arc_tigers.utils import get_configs, load_yaml

logger = logging.getLogger(__name__)


def main(args):
    # Load experiment configuration
    # and set up the experiment directory
    exp_config = load_yaml(args.exp_config)
    data_config, model_config = get_configs(exp_config)
    exp_name = args.exp_name if args.exp_name else exp_config["exp_name"]
    save_dir = (
        f"outputs/{data_config['data_name']}"
        f"/{data_config['data_args']['setting']}"
        f"/{data_config['data_args']['target_config']}"
        f"/{model_config['model_id']}/{exp_name}"
    )
    if os.path.exists(save_dir):
        base_save_dir = save_dir
        counter = 1
        while os.path.exists(save_dir) and counter < 5:
            save_dir = f"{base_save_dir}_{counter}"
            counter += 1

    os.makedirs(save_dir, exist_ok=False)

    # Save experiment configuration to the save directory
    exp_config[save_dir] = save_dir
    exp_config_path = os.path.join(save_dir, "experiment_config.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=4)
    print(f"Experiment configuration saved to {exp_config_path}")

    # set up logging
    logging.basicConfig(filename=f"{save_dir}/logs.log", level=logging.INFO)

    # Load model
    pipeline = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(
            **model_config["model_kwargs"], random_state=exp_config["seed"]
        ),
    )

    # load dataset
    train_dataset, eval_dataset, _, _ = get_reddit_data(
        **data_config["data_args"], tokenizer=None
    )

    # Train the model
    pipeline.fit(train_dataset["text"], train_dataset["label"])

    # Evaluate the model
    eval_predictions = pipeline.predict(eval_dataset["text"])
    eval_results = tfidf_evaluation(eval_dataset["label"], eval_predictions)
    print("Evaluation results:", eval_results)
    # Save evaluation results to a JSON file
    results_path = f"{save_dir}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Evaluation results saved to {results_path}")

    # Save the model and tokenizer
    joblib.dump(pipeline, f"{save_dir}/tfidf_logreg_pipeline.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument(
        "exp_config",
        help="path to the data config yaml file",
    )
    parser.add_argument(
        "--exp_name",
        help="override the experiment name in the config file",
        default=None,
    )
    args = parser.parse_args()
    main(args)
