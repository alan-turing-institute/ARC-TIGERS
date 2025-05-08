import argparse
import json
import logging
import os

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from arc_tigers.eval.utils import compute_metrics
from arc_tigers.training.utils import (
    WeightedLossTrainer,
    get_label_weights,
    get_reddit_data,
)
from arc_tigers.utils import load_yaml

logger = logging.getLogger(__name__)


def main(args):
    data_config = load_yaml(args.data_config)
    model_config = load_yaml(args.model_config)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=False)
    exp_config = {
        "data_config_pth": args.data_config,
        "model_config_pth": args.model_config,
        "save_dir": save_dir,
        "seed": 42,
    }
    # Save experiment configuration to the save directory
    exp_config_path = os.path.join(save_dir, "experiment_config.json")
    with open(exp_config_path, "w") as f:
        json.dump(exp_config, f, indent=4)
    print(f"Experiment configuration saved to {exp_config_path}")
    # set up logging
    logging.basicConfig(filename=f"{save_dir}/logs.log", level=logging.INFO)

    # Load tokenizer and model
    model_name = model_config["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_config["model_kwargs"]
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset, eval_dataset, _ = get_reddit_data(
        **data_config["data_args"], tokenizer=tokenizer
    )

    training_args = TrainingArguments(
        output_dir=save_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=f"{save_dir}/logs",
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
    for key in ["eval_f1", "eval_precision", "eval_recall"]:
        if key in results:
            results[key] = results[key].tolist()
    print("Evaluation results:", results)
    # Save evaluation results to a JSON file
    results_path = f"{save_dir}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation results saved to {results_path}")

    # Save the model
    model.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument(
        "data_config",
        help="path to the data config yaml file",
    )
    parser.add_argument(
        "model_config",
        help="path to the model config yaml file",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        default=None,
        help="Path to save the model and results",
    )
    args = parser.parse_args()

    main(args)
