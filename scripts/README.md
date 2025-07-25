# Running Experiments with ARC-TIGERS
This explains how to generate datasets and train/evaluate classifiers using the code in this repository.

## 1. Generating Data

First a dataset should be generated, this is done by first running `data_processing/dataset_download.py` if there isn't already data in your repository.

### `dataset_download.py`

This downloads data and saves it in the `data` directory, it is used like below. Eg. downloading 10000000 rows from the top subreddits defined in `data/top_subreddits.json`:
```
python scripts/data_processing/dataset_download.py
    --max_rows 10000000
    --target_subreddits data/top_subreddits.json
```
More information can be found in `scripts/data_processing/README.md`

**At this stage, the raw data has been downloaded from the HuggingFace Datasets page, and we have filtered out the subreddits we don't want**

### `dataset_generation.py`

This generates the actual dataset, i.e. the CSV files used in our experiments.

Eg. Generate one-vs-all splits for the "football" configuration, using the 10000000 rows downloaded above:
```
python scripts/data_processing/dataset_generation.py data/reddit_dataset_12/10000000_rows/ football --mode one-vs-all
```

**At this stage, the data has from the previous stage been filtered to remove empty rows and '[removed]' or '[deleted]' posts. We then partition the dataset according to a specified split strategy defined in `arc_tigers/data/reddit_data.py`. These splits are then saved to `train.csv` and `test.csv`**

To be used in the next step you should now specify data configs which are defined like so:

```yaml
data_name: reddit_dataset_12
data_args:
  n_rows: 10000000
  task: one-vs-all
  target_config: football
  balanced: True
```

The `data_args` component of these configs are passed to [`get_reddit_data`](https://github.com/alan-turing-institute/ARC-TIGERS/blob/30-one-vs-all-classification/src/arc_tigers/data/reddit_data.py#L110) in `arc_tigers/data/reddit_data.py` to load the data. `data_name` is used for naming saved outputs.

For more information on `get_reddit_data` see [`arc_tigers/data/README.md`](https://github.com/alan-turing-institute/ARC-TIGERS/blob/30-one-vs-all-classification/src/arc_tigers/data/README.md).

## 2. Training models

This is for training models on the generated datasets, for this you need to pass `train_classifier.py` an experiment config, which is composed of a `data_config`, a `model_config`, and training arguments:

```
python scripts/experiments/train_classifier.py configs/experiment/football_one_vs_all_balanced.yaml
```

Where the yaml might be structured like so:
``` yaml
model_config: distilbert
data_config: football_one_vs_all_balanced
exp_name: football_distilbert_one_vs_all
seed: 42
train_kwargs:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  warmup_ratio: 500
  weight_decay: 0.01
  save_total_limit: 3
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end : True
```
Once your model is trained it will be saved in an `outputs` directory, in an appropriate subdirectory, for example above will be saved to:
- `outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all
`

Where it is saved under the name of the data, the evaluation setting, the data config, then the model, and finally the experiment config.

### Notes
- Model configs are defined as such (eg.`distilbert`):
  ```yaml
  model_id: "distilbert-base-uncased"
  model_kwargs:
    num_labels: 2
  ```
- In this script data is loaded using the [`get_reddit_data`](https://github.com/alan-turing-institute/ARC-TIGERS/blob/30-one-vs-all-classification/src/arc_tigers/data/reddit_data.py#L110) function.

- Note: The experiment name can be overwritten when running  `train_classifier.py` using an optional argument: `--exp_name`.

## 3. Running sampling strategies

You should now have a trained transformers model saved in an appropriately convoluted directory structure. Let's evaluate it!

You can run sampling using the distance-based strategy like so:

```
python scripts/experiments/active_testing.py configs/data/football_one_vs_all_balanced.yaml configs/model/distilbert.yaml outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all distance --n_repeats 10 --max_labels 500 --seed 42
```
Where the arguments are the `<data_config>`, the `<model_config>`, the directory where the above model should be saved, using the distance-based sampling strategy. We also perform 10 repeats with a labelling budget of 500.

## 4. Analysing results

You should now rapidly be approaching your path length limit for your chosen operating system. To make matter worse we will now generate some figures... saved in another subdirectory... `figures`.

To get plots/analysis for an individual model you can run `scripts/evaluation/eval_sampling.py`.

For example, to get the random sampling outputs from the above experiment:

```
python scripts/evaluation/eval_sampling.py outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all/eval_outputs/football_one_vs_all_balanced/random_sampling_outputs
```

If you wish to compare multiple different configurations you can run `scripts/evaluation/sampling_error.py`, this will produce figures comparing the mean squared error of different sampling strategies to the baseline value.

It can be run like below. For exampling, comparing distance-based sampling to random sampling:

```
python scripts/evaluation/sampling_error.py outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all/ sport_one_vs_all_balanced --experiments <full_path_to_random_sampling_output_folder> <full_path_to_distance_based_sampling_output_folder>
```
