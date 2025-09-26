# Data Scripts

Use `--help` for each script to see arguments.

## `dataset_download.py`

This script downloads a Reddit dataset (via HuggingFace Datasets), filters it to include only specified subreddits, and saves a subset of rows from specified sub-reddits.

Outputs saved to: `data/<dataset_name>`

## `dataset_generation.py`

This script processes filtered Reddit data and generates train/test splits for a specific data config. See [`../../configs/data`](../../configs/data) for example data configs.

Outputs saved to: `data/<dataset_name>/splits/<seed>_<train_imbalance>`
