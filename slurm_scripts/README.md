# `slurm_scripts`
This directory contains bash scripts for the `Slurm` workload manager, they are split into their corresponding tasks in this codebase.

One liner for getting a slurm job output:

```
tail -f $(scontrol show job <JOBID> | awk -F= '/StdOut/ {print $2}')
```

## `data_download_scripts`

This sub-directory contains scripts necessary for downloading the data for our experiments.

### `generate_reddit_dataset.sh`

This script downloads a specified number of rows from the the `bit0/reddit_dataset_12` dataset from huggingface, and also optionally generates a split from `ONE_VS_ALL_COMBINATIONS` of `data/reddit_data.py`. To do this pass the specified number of rows as input argument 1 (`$1`) and optionally pass the specified split as the second input argument (`$2`), eg:

```
sbatch slurm_scripts/data_download_scripts/generate_reddit_dataset.sh 100000 sport
```
