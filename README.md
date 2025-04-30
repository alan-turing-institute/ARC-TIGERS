# ARC-TIGERS

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Testing Imbalanced cateGory classifiERS

## Installation

```bash
python -m pip install arc_tigers
```

From source:
```bash
git clone https://github.com/alan-turing-institute/ARC-TIGERS
cd ARC-TIGERS
python -m pip install .
```

## Usage

### `scripts/dataset_download.py`
This script downloads a reddit dataset and saves in an appropriate place in the parent directory.
It takes as arguments:
- `dataset_name`: the name of the dataset to load from huggingface, for example `bit0/reddit_dataset_12`
- `target_subreddits`: a list subreddits being used for the experiment, this should be in `.json` format.
- `max_rows`: The maximum number of rows to use in the resultant dataset, this should be an integer.
It saves a `.json` file called `filtered_rows` containing the data in a subdirectory named using the dataset name and the maximum number of rows.

### `scripts/dataset_generation.py`
This script generates train and test splits from the downloaded reddit dataset(s).
It currently takes as arguments:
- `data_dir`: the path to the dataset being used to form the splits
- `split`: the specific subreddit split to generate. Defined in the dictionary `DATASET_COMBINATIONS` in `/data/utils`.
- `r`: The ratio of target subreddits to non-target subreddits

The script saves two csv files, `train.csv` and `test.csv` within a subdirectory `splits`, these contain the train and evaluation splits and are of roughly equal size.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-TIGERS/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-TIGERS/actions
[pypi-link]:                https://pypi.org/project/ARC-TIGERS/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/ARC-TIGERS
[pypi-version]:             https://img.shields.io/pypi/v/ARC-TIGERS
<!-- prettier-ignore-end -->
