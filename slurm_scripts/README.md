# `slurm_scripts`
This directory contains bash scripts for the `Slurm` workload manager, they are split into their corresponding tasks in this codebase.

One liner for getting a slurm job output:

```
tail -f $(scontrol show job <JOBID> | awk -F= '/StdOut/ {print $2}')
```
This can be ran using `show_job_output.sh` like so:
```
slurm_scripts/show_job_output.sh <job_id>
```

## `data_download_scripts`

This sub-directory contains scripts necessary for downloading the data for our experiments.

### `generate_reddit_dataset.sh`

This script downloads a specified number of rows from the the `bit0/reddit_dataset_12` dataset from huggingface, and also optionally generates a split from `ONE_VS_ALL_COMBINATIONS` of `data/reddit_data.py`. To do this pass the specified number of rows as input argument 1 (`$1`) and optionally pass the specified split as the second input argument (`$2`), eg:

```
sbatch slurm_scripts/data_download_scripts/generate_reddit_dataset.sh 100000 sport
```

Pass a third argument to specify the split mode, that is: `data-drift`, `one-vs-all`, or `multi-class`.

## `training_scripts`
This sub-directory contains training scripts

### `train_hf_classifier.sh`

This script launches a training job for a HuggingFace classifier using a specified YAML configuration file. It sets up the environment, loads required modules, and runs the training script with the provided configuration.

#### Usage

Submit the job to Slurm with:

```
sbatch slurm_scripts/training_scripts/train_hf_classifier.sh <CONFIG_NAME>
```

Where `<CONFIG_NAME>` is the name of the YAML config file (without the `.yaml` extension) located in `configs/training/`. For example:

```
sbatch slurm_scripts/training_scripts/train_hf_classifier.sh football_one_vs_all_1_in_10
```

This will run:

```
python scripts/experiments/train_classifier.py configs/training/football_one_vs_all_1_in_10.yaml
```