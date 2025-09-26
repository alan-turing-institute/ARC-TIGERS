# Experiments

Use `--help` to see arguments for these scripts.

## Model Training

### `train_classifier.py`

This script trains a text classifier on Reddit data using HuggingFace Transformers, using a training config (see [`../../configs/training`](../../configs/training)), model config (see [`../../configs/model`](../../configs/model)), and hyperparameters config (see [`../../configs/hparams`](../../configs/hparams)).

Outputs saved to: `outputs/<dataset_name>/<task_name>/<target_name>/<seed>_<train_imbalance>/<model_name>/<hparams_name>`

### `train_tfidf_classifier.py`

The same as `train_classifier.py`, but trains a TF-IDF classifier instead.

Outputs saved to: `outputs/<dataset_name>/<task_name>/<target_name>/<seed>_<train_imbalance>/<model_name>/<hparams_name>`

## Sampling

### `sample.py`

This script iteratively selects data to label from a dataset using one of several strategies (the supported strategies are `SAMPLING_STRATEGIES` in [`src/arc_tigers/samplers/methods.py`](../../src/arc_tigers/samplers/methods.py)).

Outputs saved to: `outputs/<dataset_name>/<task_name>/<target_name>/<seed>_<train_imbalance>/<model_name>/<hparams_name>/<eval_outputs>/<test_imbalance>/<sampling_strategy>`

### `sample_replay.py`

The script uses the order of the data chosen to label from one sampling strategy & model to evaluate a different model.

Outputs saved to: `outputs/<dataset_name>/<task_name>/<target_name>/<seed>_<train_imbalance>/replays/<test_imbalance>/<eval_model_name>_<hparams_name>/from_<model_name>_<hparams_name>/<sampling_strategy>`

(where `<eval_model_name>` is the name of the model being evaluated, and `from_<model_name>` is the name of the model used to create the labelled subset)
