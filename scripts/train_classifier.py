from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from arc_tigers.data.utils import preprocess_function
from arc_tigers.eval.utils import compute_metrics

# Load tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# load dataset
dataset = load_dataset("csv", data_files={"train": "train.csv", "eval": "test.csv"})


tokenized_datasets = dataset.map(
    preprocess_function, batched=True, fn_kwargs={"tokenizer": tokenizer}
)

# Split dataset
train_dataset = tokenized_datasets["train"]
train_dataset, test_dataset = train_dataset.train_test_split(test_size=0.1).values()

eval_dataset = tokenized_datasets["eval"]

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./sentiment-classifier")
tokenizer.save_pretrained("./sentiment-classifier")
