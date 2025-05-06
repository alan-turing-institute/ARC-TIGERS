import argparse
import json
import os
from collections import Counter

from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
)

from arc_tigers.data.utils import (
    BINARY_COMBINATIONS,
    ONE_VS_ALL_COMBINATIONS,
    get_target_mapping,
    preprocess_function,
)
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.training.utils import WeightedLossTrainer


def main(args):
    setting = args.setting
    split = args.split
    balanced = args.balanced
    data_dir = f"{args.data_dir}/splits/{split}/"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=False)

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # load dataset
    dataset = load_dataset(
        "csv",
        data_files={"train": f"{data_dir}/train.csv", "eval": f"{data_dir}/test.csv"},
    )

    if setting == "multi-class":
        targets = BINARY_COMBINATIONS[split]["train"]
        targets = ["r/soccer", "r/FantasyPL"]
        dataset = dataset.filter(lambda y: y in targets, input_columns=["label"])
        target_map = get_target_mapping(setting, targets)

    elif setting == "one-vs-all":
        targets = ONE_VS_ALL_COMBINATIONS[split]["train"]
        target_map = get_target_mapping(setting, targets)

    tokenized_datasets = dataset.map(
        preprocess_function,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "targets": target_map,
        },
    )

    # balance the dataset
    if balanced:
        print("Balancing the dataset...")
        # Get the number of samples in each class
        label_counts = Counter(tokenized_datasets["train"]["label"])
        # Calculate the weights for each class
        min_samples = min(label_counts.values())
        resampled_data = []
        for label in label_counts:
            label_data = tokenized_datasets["train"].filter(
                lambda x, label=label: x["label"] == label
            )
            resampled_data.append(label_data.shuffle().select(range(min_samples)))
        tokenized_datasets["train"] = concatenate_datasets(resampled_data)

    # Split dataset
    train_data = tokenized_datasets["train"]
    train_dataset, eval_dataset = train_data.train_test_split(test_size=0.1).values()

    print("Label counts in the training dataset:")
    label_counter = Counter(train_dataset["label"])
    label_counts = dict(sorted(label_counter.items(), key=lambda item: item[1]))
    label_weights = [
        1 - (label_count / sum(label_counts.values()))
        for label_count in label_counts.values()
    ]

    for label_idx, (label, count) in enumerate(label_counts.items()):
        print(f"Label: {label}, Count: {count}, Weight: {label_weights[label_idx]}")

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
        loss_weights=label_weights,
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
