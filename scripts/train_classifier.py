from collections import Counter

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from arc_tigers.data.utils import (
    BINARY_COMBINATIONS,
    ONE_VS_ALL_COMBINATIONS,
    get_target_mapping,
    preprocess_function,
)
from arc_tigers.eval.utils import compute_metrics
from arc_tigers.utils import get_device

device = get_device()
# Ensure model is moved to the selected device

setting = "multi-class"
target_subreddits = ["r/soccer", "r/Cricket"]

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

# Split dataset
train_data = tokenized_datasets["train"]
train_dataset, test_dataset = train_data.train_test_split(test_size=0.1).values()

eval_dataset = tokenized_datasets["eval"]

label_counts = Counter(train_dataset["label"])
print("Label counts in the training dataset:")
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")

# Trainer
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    output_directory="../sentiment-classifier",
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("../sentiment-classifier")
tokenizer.save_pretrained("../sentiment-classifier")
