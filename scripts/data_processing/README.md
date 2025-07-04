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

This script processes filtered Reddit data and generates train/test splits for different experimental settings (one-vs-all, data-drift, multi-class). It supports both single JSON files and sharded datasets.

## Usage

### Arguments:

- `data_dir` (required):
  - Path to the data directory or JSON file containing filtered Reddit data.
  -
- `target_config` (required): The target configuration to use for the split. Options include: sport, american_football, ami, news, advice.

- `--mode` (default: `one-vs-all`):
  - Splitting mode. Options: one-vs-all, data-drift, multi-class.

- `-r` (optional):
  - Imbalance ratio for the dataset. For example, 0.01 means 1% of the dataset belongs to the target/positive class; 100 means 100 samples belong to the target/positive class.

### Output:

Train and test splits are saved in: splits/<target_config>_<mode>/train.csv and test.csv (or as sharded files if input is sharded).

Example usage:

Generate one-vs-all splits for the "football" configuration:
```
python dataset_generation.py data/reddit_dataset_12/10000000_rows/ football --mode one-vs-all
```

Generate data-drift splits for the "news" configuration:
```
python dataset_generation.py data/reddit_dataset_12/10000000_rows/ news --mode data-drift
```

## Notes:

- The script automatically detects whether the input is a directory of shards or a single JSON file.
- Output directories are created automatically.
- For available target_config options and more details, see the code and comments in `src/arc_tigers/data/reddit_data.py`.
