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

from arc_tigers.data.utils import hf_train_test_split, tokenize_data
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.training.config import TrainConfig
from arc_tigers.training.utils import WeightedLossTrainer, get_label_weights
from arc_tigers.utils import seed_everything

logger = logging.getLogger(__name__)


def main(args):
    train_config = TrainConfig.from_path(args.train_config)

    save_dir = train_config.save_dir
    os.makedirs(save_dir, exist_ok=False)

    logging.basicConfig(filename=f"{save_dir}/logs.log", level=logging.INFO)

    train_config.save(save_dir)
    logger.info("Experiment configuration saved to %s", save_dir)
    seed = train_config.hparams_config.train_kwargs["seed"]
    seed_everything(seed)

    # Load tokenizer and model
    model_name = train_config.model_config.model_id
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    model: PreTrainedModel = AutoModelForSequenceClassification.from_pretrained(
        model_name, **train_config.model_config.model_kwargs
    )

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = train_config.data_config.get_train_split()
    train_dataset = tokenize_data(train_dataset, tokenizer)
    train_dataset, eval_dataset = hf_train_test_split(
        train_dataset, seed=seed, test_size=0.1
    )

    training_args = TrainingArguments(
        output_dir=str(save_dir),
        logging_dir=f"{save_dir}/logs",
        **train_config.hparams_config.train_kwargs,
    )
    # Trainer
    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        loss_weights=get_label_weights(train_dataset)[0],
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    logger.info("Evaluation results: %s", results)
    # Save evaluation results to a JSON file
    results_path = save_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Evaluation results saved to %s", results_path)

    # Save the model and tokenizer
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument("train_config", help="path to the train config yaml file")
    args = parser.parse_args()
    main(args)
