import argparse
import json
import logging
import os

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from arc_tigers.data.reddit_data import get_reddit_data
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.training.utils import WeightedLossTrainer, get_label_weights
from arc_tigers.utils import get_configs, load_yaml

logger = logging.getLogger(__name__)


def main(args):
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
    exp_config[save_dir] = save_dir
    # Save experiment configuration to the save directory
    exp_config_path = os.path.join(save_dir, "experiment_config.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=4)
    print(f"Experiment configuration saved to {exp_config_path}")
    # set up logging
    logging.basicConfig(filename=f"{save_dir}/logs.log", level=logging.INFO)

    # Load tokenizer and model
    model_name = model_config["model_id"]
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_config["model_kwargs"]
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset, eval_dataset, _, _ = get_reddit_data(
        **data_config["data_args"], tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        logging_dir=f"{save_dir}/logs",
        **exp_config["train_kwargs"],
    )
    # Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        loss_weights=get_label_weights(train_dataset),
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print("Evaluation results:", results)
    # Save evaluation results to a JSON file
    results_path = f"{save_dir}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")

    # Save the model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


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
