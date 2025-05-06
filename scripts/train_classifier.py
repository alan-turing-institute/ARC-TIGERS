import argparse
import json
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


def main(args):
    target_config = args.target_config
    data_dir = f"{args.data_dir}/splits/{target_config}/"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=False)

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset, eval_dataset = get_reddit_data(
        setting=args.setting,
        target_config=target_config,
        balanced=args.balanced,
        data_dir=data_dir,
        tokenizer=tokenizer,
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
        "--balanced",
        action="store_true",
        help="Whether to balance the dataset",
    )
    parser.add_argument(
        "--setting",
        type=str,
        choices=["multi-class", "one-vs-all"],
        default="multi-class",
        help="Evaluation setting",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="sport",
        help="Split to use for training",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/reddit_dataset_12/15000000_rows/",
        help="Path to the data used for training",
    )
    parser.add_argument(
        "save_dir",
        type=str,
        default=None,
        help="Path to save the model and results",
    )
    args = parser.parse_args()

    main(args)
