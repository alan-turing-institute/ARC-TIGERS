# ARC-TIGERS

[![Actions Status][actions-badge]][actions-link]

**T**esting **I**mbalanced cate**G**ory classifi**ERS**.

This project explored several approaches for estimating performance metrics on a dataset with as few samples from it as possible.

## Installation

```bash
git clone https://github.com/alan-turing-institute/ARC-TIGERS
cd ARC-TIGERS
uv sync
```

## Usage

The repo is set up with roughly the following main steps:

1. Dataset generation (see [`scripts/data_processing`](scripts/data_processing)): Download a Reddit dataset and use it to create several splits of it for subreddit classification tasks with varying levels of imbalance.
2. Model training (see [`scripts/experiments`](scripts/experiments)): Fine-tune classifiers on the Reddit datasets.
3. Sampling (label subset selection, see [`scripts/experiments`](scripts/experiments)): Use a strategy to iteratively decide which data point from a test set should be selected to label next to give the best possible estimate of a range of performance metrics.
4. Evaluation of a single experiment (repeated runs of the same sampling config with different seeds), with the goal of the estimated metric value being as close to the ground truth as possible. See [`scripts/evaluation`](scripts/evaluation)
5. Analysis aggregating results across many experiments: See [`scripts/analysis`](scripts/analysis)

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-TIGERS/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-TIGERS/actions
<!-- prettier-ignore-end -->
