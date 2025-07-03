# `train_classifier.py`

This script trains a text classifier on Reddit data using HuggingFace Transformers, using an experiment config.

## Usage:
### Arguments:

- `exp_config` (required):
  - Path to the YAML config file specifying model, data, and training parameters.
- `--exp_name` (optional):
  - Custom experiment name to override the one in the config.

### What the script does:

- Loads the experiment configuration from YAML.
- Sets up output directories and logging.
- Loads the tokenizer and model specified in the config.
- Loads and preprocesses the Reddit dataset according to the config.
- Sets up the HuggingFace Trainer with weighted loss (for class imbalance).
- Trains the model and saves checkpoints, logs, and the final model.
- Evaluates the model and saves evaluation metrics to a JSON file.

### Output:

All outputs (logs, checkpoints, evaluation results, and the trained model) are saved in a directory under `outputs/` based on your configs and experiment name.

### Example:

```
python train_classifier.py football_one_vs_all_balanced.yaml
```

### Notes:

- You can adjust training parameters (epochs, batch size, learning rate, etc.) in your YAML config file.
- For available dataset settings and target configs, see [`get_reddit_data`](https://github.com/alan-turing-institute/ARC-TIGERS/blob/d40b20bc876e31ee58beadbef4f83b18d883366c/src/arc_tigers/data/reddit_data.py#L110) in `src/arc_tigers/data/reddit_data.py`
- For custom loss weighting, see [`WeightedLossTrainer`](https://github.com/alan-turing-institute/ARC-TIGERS/blob/d40b20bc876e31ee58beadbef4f83b18d883366c/src/arc_tigers/training/utils.py#L10) in `src/arc_tigers/training/utils.py`


# `sample.py`

This script runs active testing experiments using various acquisition strategies to iteratively select and label samples from a dataset. It supports the following strategies: uniform random sampling, distance-based, accuracy-based, cross entropy based, isolation forest, and minority class sampling. The script evaluates model performance as more samples are labeled, saving metrics and statistics at each step. Uniform random sampling can also be run with synthetic data and models.

## Arguments:

- `<data_config.yaml>`:
    - Path to the data config YAML file
- `<model_config.yaml>`:
    - Path to the model config YAML file
- `<save_dir>`:
  - Directory to save results and outputs.
- `<acq_strat>`:
  - Acquisition strategy to use. Options: random, distance, acc_sampler, ig_sampler, isolation_forest, minority_class
- `--n_repeats`:
  - Number of experiment repetitions.
- `--max_labels`:
  - Maximum number of samples to label in each run.
- `--seed`:
  - Random seed for reproducibility.
- `--eval_every`:
  - (Optional) Evaluate metrics every K samples (default: 50).
- `--min_labels`:
  - (Optional) Minimum number of labels before computing metrics (default: 10).
- `--class_balance`:
  - (Optional) Class balance ratio (default: 1.0).

## What the script does:

- Loads or generates predictions for the evaluation dataset.
- Sets up the chosen acquisition function (sampling strategy).
- Computes and saves full-dataset metrics and statistics.
- Iteratively samples and labels data, evaluating and saving metrics at each step, repeated for the specified number of runs.
- Saves all metrics and statistics as CSV and JSON files in the output directory.

## Output:

- Metrics for the full dataset:
  - `metrics_full.json`, `stats_full.json`
- Iterative metrics for each run:
  - `metrics_<subsampling_seed>`.csv

All outputs are saved at `<save_dir>/eval_outputs/<data_config>/` under an appropriately named subdirectory, depending on the data being used for evaluation.

## Example:
```
python sample.py football_one_vs_all_balanced.yaml distilbert.yaml outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all distance --n_repeats 10 --max_labels 500 --seed 42
```

## Notes:

- Acquisition strategies supported: random, distance, acc_sampler, ig_sampler, isolation_forest, minority_class. For reference see `arc_tigers/sample/acquisition.py` and `arc_tigers/sample/random.py`.
