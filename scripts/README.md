# Running Experiments with ARC-TIGERS
This explains how to generate datasets and train/evaluate classifiers using the code in this repository.

## 1. Generating Data

First a dataset should be generated, this is done by first running `data_processing/dataset_download.py` if there isn't already data in your repository.

### `dataset_download.py`

This downloads data and saves it in the `data` directory, it us used like below. Eg. downloading 10000000 rows from the top subreddits defined in `data/top_subreddits.json`:
```
python scripts/data_processing/dataset_download.py
    --max_rows 10000000
    --target_subreddits data/top_subreddits.json
```
More information can be found in `scripts/data_processing/README.md`

### `dataset_generation.py`

This generates the actual dataset, csv files used in our experiments.

Eg. Generate one-vs-all splits for the "football" configuration, using the 10000000 rows downloaded above:
```
python dataset_generation.py data/reddit_dataset_12/10000000_rows/ football --mode one-vs-all
```

You should now specify data configs which are defined like so:

```yaml
data_name: reddit_dataset_12
data_args:
  n_rows: 10000000
  setting: one-vs-all
  target_config: football
  balanced: True
```

## 2. Training models

This is for training models on the generated datasets, for this you need to pass `train_classifier.py` an experiment config, which is composed of a `data_config` a `model_config` and training arguments:

```
python train_classifier.py football_one_vs_all_balanced.yaml
```

Where the yaml might be structured like so:
``` yaml
model_config: distilbert
data_config: football_one_vs_all_balanced
exp_name: football_distilbert_one_vs_all
random_seed: 42
train_kwargs:
  num_train_epochs: 3
  per_device_train_batch_size: 8
  per_device_eval_batch_size: 8
  warmup_steps: 500
  weight_decay: 0.01
  save_total_limit: 3
  eval_strategy: epoch
  save_strategy: epoch
  load_best_model_at_end : True
```
Once your model is trained it will be saved in an `outputs` directory, in an appropriate subdirectory, for example above will be saved to:
- `outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all
`

Where it is saved under the name of the data, the evaluation setting, the data config,  then the model, and finally the experiment config.

### Notes
- Model configs are defined as such (eg.`distilbert`):
  ```yaml
  model_id: "distilbert-base-uncased"
  model_kwargs:
    num_labels: 2
  ```
- In this script data is loaded using the [`get_reddit_data`](https://github.com/alan-turing-institute/ARC-TIGERS/blob/d40b20bc876e31ee58beadbef4f83b18d883366c/src/arc_tigers/data/reddit_data.py#L110) function.

- Note: The experiment name can be overwritten when running  `train_classifier.py` using an optional argument: `--exp_name`.

## 3. Running sampling strategies

You should now have a trained transformers model saved in an appropriately convoluted directory structure. Let's evaluate it!

You can run active testing like so:

```
python active_testing.py football_one_vs_all_balanced.yaml distilbert.yaml outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all distance --n_repeats 10 --max_labels 500 --seed 42
```
Where the arguments are the `<data_config>`, the `<model_config>`, the directory where the above model should be saved, using the distance-based sampling strategy. We also perform 10 repeats with a labelling budget of 500.


Random sampling is run similarly with identical format at the command line to above, only running `random_sampling.py` instead and omitting the sampling strategy. Eg.:

```
python random_sampling.py football_one_vs_all_balanced.yaml distilbert.yaml outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all --n_repeats 10 --max_labels 500 --seed 42
```

The outputs are saved similarly to `active_testing.py`.

## 4. Analysing results

You should now rapidly be approaching your path length limit for your chosen operating system. To make matter worse we will now generate some figures.. saved in another subdirectory.. `figures`.

To get plots/analysis for an individual model you can run `eval_sampling.py`.

For example, to get the random sampling outputs from the above experiment:

```
python eval_sampling.py outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all/eval_outputs/football_one_vs_all_balanced/random_sampling_outputs
```

If you wish to compare multiple different configurations you can run `sampling_error.py`, this will produce figures comparing the mean squared error of different sampling strategies to the baseline value.

It can be run like below. For exampling, comparing distance-based sampling to random sampling:

```
python sampling_error.py outputs/reddit_dataset_12/one-vs-all/football/distilbert-base-uncased/football_distilbert_one_vs_all/ sport_one_vs_all_balanced --experiments <full_path_to_random_sampling_output_folder> <full_path_to_distance_based_sampling_output_folder>
```
