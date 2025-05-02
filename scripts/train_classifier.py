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
from arc_tigers.utils import get_device

device = get_device()
# Ensure model is moved to the selected device

balanced = False

setting = "multi-class"

split = "sport"

data_dir = f"../data/reddit_dataset_12/15000000_rows/splits/{split}/"
# data_dir = f"../data/reddit_dataset_12/1500_rows/splits/{split}/"

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)


# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load dataset
dataset = load_dataset(
    "csv", data_files={"train": f"{data_dir}/train.csv", "eval": f"{data_dir}/test.csv"}
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

test_dataset = tokenized_datasets["eval"]

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
    output_dir="../sentiment-classifier",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="../sentiment-classifier/logs",
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

# Save the model
model.save_pretrained("../sentiment-classifier")
tokenizer.save_pretrained("../sentiment-classifier")
