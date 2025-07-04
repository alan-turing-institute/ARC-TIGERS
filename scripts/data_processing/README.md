# `dataset_download.py`

This script downloads a Reddit dataset (via HuggingFace Datasets), filters it to include only specified subreddits, and saves a subset of rows from specified sub-reddits.

## Usage
### Arguments
- `--max_rows (int, required)`:
  - Maximum number of rows to process and save.

- `--dataset_name (str, default: bit0/reddit_dataset_12)`:
    - Name of the HuggingFace dataset to load.

- `--target_subreddits (str, default: top_subreddits.json)`:
    - Path to a JSON file containing a list of target subreddits.

- `--seed (int, default: 42)`:
  - Random seed for data shuffling.

### Output
The filtered data will be saved in: `DATA_DIR/<dataset_name_out>/<max_rows>_rows/`, where `<dataset_name_out>` is `reddit_dataset_12` by default.

### Example
Download 1,000,000 rows from the default dataset, filtering for subreddits in top_subreddits.json, and save as a single file:

```
python scripts/data_processing/dataset_download.py
    --max_rows 1000000
    --target_subreddits data/top_subreddits.json
```
Download 10,000,000 rows, sharded into multiple files:

```
python scripts/data_processing/dataset_download.py
    --max_rows 10000000
    --sharding True
```

## Notes
- The script uses streaming mode to avoid loading the entire dataset into memory.
- Make sure your `target_subreddits` JSON file contains a list of subreddit names (case-insensitive).
- Output directory is created automatically if it does not exist.


# `dataset_generation.py`

This script processes filtered Reddit data and generates train/test splits for different experimental settings (one-vs-all, data-drift, multi-class) and levels of imbalance.

## Usage

### Arguments:

- `data_config` (required):
  - Path to the data config to use.

### Output:

Train and test splits are saved in: <dataset_name>/splits/<data_config_name>.

Example usage:

```
python dataset_generation.py configs/data/football_one_vs_all_1_in_10.yaml
```
