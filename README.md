# ARC-TIGERS

[![Actions Status][actions-badge]][actions-link]

Testing Imbalanced cateGory classifiERS.

## Installation

```bash
git clone https://github.com/alan-turing-institute/ARC-TIGERS
cd ARC-TIGERS
uv sync
```

## Usage

1. Dataset generation: See [`scripts/data_processing`](scripts/data_processing)
2. Model training: See [`scripts/experiments`](scripts/experiments)
3. Sampling (label subset selection): See [`scripts/experiments`](scripts/experiments)
4. Evaluation of a single experiment (repeated runs of the same sampling config with different seeds): See [`scripts/evaluation`](scripts/evaluation)
5. Analysis aggregating results across many experiments: See [`scripts/analysis`](scripts/analysis)

## License

Distributed under the terms of the [MIT license](LICENSE).


<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/alan-turing-institute/ARC-TIGERS/workflows/CI/badge.svg
[actions-link]:             https://github.com/alan-turing-institute/ARC-TIGERS/actions
<!-- prettier-ignore-end -->
