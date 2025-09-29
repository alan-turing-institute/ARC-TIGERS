# Analysis Scripts

This directory contains scripts for analysing and aggregating results across multiple experiments. The scripts operate on the outputs from individual experiments and provide higher-level insights into model performance, sampling strategy effectiveness, and cross-experiment comparisons

## Scripts Description

### `aggregate_analysis.py`
Aggregates results across multiple experiments to determine which sampling strategies perform better than random baseline. Computes percentage improvements and generates LaTeX tables.

**Usage:**
```bash
python scripts/analysis/aggregate_analysis.py \
    outputs/reddit_dataset_12/one-vs-all/football/42_05 \
    --models distilbert gpt2 ModernBERT \
    --imbalances 05 01 001 \
    --strategies accuracy_lightgbm info_gain_lightgbm minority ssepy isolation \
    --metrics accuracy f1_score precision recall
```

**Parameters:**
- `base_path` (required): Base experiment directory path
- `--models, -m` (required): List of model names to analyse
- `--imbalances, -i` (required): Test imbalance levels (e.g., '05', '01', '001')
- `--strategies, -s` (required): Sampling strategies to evaluate
- `--metrics, -mt` (optional): Metrics to compare (defaults to all available metrics from random baseline)

**Description:**
- Iterates over combinations of models, imbalances, and strategies
- Generates LaTeX tables in `tmp/tables/aggregate_performance/`
- Computes average improvements across models for each configuration
- Uses sample sizes: 110, 260, 510, 1000

### `compare_models.py`
Compares performance of different models using the same sampling strategy and imbalance level. Creates detailed visualizations comparing model performance.

**Usage:**
```bash
python scripts/analysis/compare_models.py \
    outputs/reddit_dataset_12/one-vs-all/football/42_05 \
    05 \
    accuracy_lightgbm \
    --models distilbert gpt2 ModernBERT zero-shot
```

**Parameters:**
- `base_path` (required): Base path like 'outputs/reddit_dataset_12/one-vs-all/football/42_05'
- `test_imbalance` (required): Test imbalance level (e.g., '05', '01', '001')
- `sampling_method` (required): Sampling method directory name (e.g., 'random', 'distance')
- `--models` (required): List of model names to compare

**Description:**
- Generates model comparison plots (raw metrics and MSE)
- Creates visualizations in organized directory structure at `{base_path}/figures/model_comparison/`
- Produces both raw performance curves and MSE comparison plots
- Automatically finds model directories matching the specified pattern
- Creates separate subdirectories for different plot types (raw/, mse/)

### `compare_sampling_strategies.py`
Compares different sampling strategies for a given model and imbalance level. Generates comprehensive visualizations comparing strategy effectiveness.

**Usage:**
```bash
python scripts/analysis/compare_sampling_strategies.py \
    outputs/reddit_dataset_12/one-vs-all/football/42_05 \
    distilbert \
    --imbalances 05 01 001 \
    --strategies random distance isolation accuracy_lightgbm info_gain_lightgbm
```

**Parameters:**
- `base_path` (required): Base path like 'outputs/reddit_dataset_12/one-vs-all/football/42_05'
- `model` (required): Model name (e.g., 'distilbert', 'gpt2', 'ModernBERT')
- `--imbalances` (required): Test imbalance levels to process (e.g., '05', '01', '001')
- `--strategies` (required): List of sampling strategies to compare

**Description:**
- Comparison of sampling strategies
- Generates three types of plots: raw metrics, MSE comparison, and improvement analysis
- Creates organized output directory structure at `{base_path}/figures/sampling_comparison/`
- Processes multiple imbalance levels

### `model_performance.py`
analyses overall model performance across different configurations.

**Description:**
- Comprehensive performance metrics analysis
- Cross-configuration performance comparison
- Statistical summaries and visualizations

### `model_performance_summary.py`
Generates summary reports of model performance across all experiments.

**Description:**
- High-level performance summaries
- LaTeX table generation for reports
- Consolidated performance metrics

### `model_transfer_analysis.py`
analyses how models perform when transferred across different imbalance levels or datasets.

**Description:**
- Label transfer performance analysis
- Produces a number of LaTeX tables

### `bootstrapping_rmse.py`
Performs bootstrap resampling analysis to assess confidence intervals for RMSE and other metrics. Provides robust statistical analysis of sampling strategy performance.

**Usage:**
```bash
python scripts/analysis/bootstrapping_rmse.py \
    outputs/reddit_dataset_12/one-vs-all/football/42_05 \
    --models distilbert gpt2 ModernBERT zero-shot \
    --imbalances 05 01 001 \
    --sampling-methods random ssepy info_gain_lightgbm accuracy_lightgbm minority isolation \
    --verbose \
    --selected-steps 10 100 500
```

**Parameters:**
- `base_path` (required): Base experiment directory path
- `--models` (optional): Model names to include (default: distilbert, ModernBERT, gpt2, zero-shot)
- `--imbalances` (optional): Test imbalance levels (default: 05, 01, 001)
- `--sampling-methods` (optional): Sampling strategies to evaluate (default: random, ssepy, info_gain_lightgbm, accuracy_lightgbm, minority, isolation)
- `--verbose, -v` (optional): Print detailed output and tables
- `--selected-steps` (optional): Only include specific evaluation steps (e.g., 10 100 500)

**Description:**
- Bootstrap confidence interval estimation with configurable sample sizes
- Statistical significance testing across all metrics
- Error estimation using bootstrap resampling
- Generates grouped analysis tables and JSON results
- Can filter to specific evaluation steps for focused analysis
- Creates tables in multiple formats: raw RMSE, differences from random, and grouped metrics
- Outputs saved to `{base_path}/tables/bootstrap_rmse/` with subdirectories for different analysis types

### `bias_correction_impact.py`
analyses the impact of bias correction techniques on sampling strategy performance.

**Description:**
- Before/after bias correction comparisons
- Impact quantification across strategies

### `aggregate_class_distribution.py`
Analyses class distribution patterns across different sampling strategies and datasets from metrics files.

**Usage:**
```bash
python scripts/analysis/aggregate_class_distribution.py \
    outputs/reddit_dataset_12/one-vs-all/football/42_05 \
    --imbalances 01 05 001 \
    --models distilbert ModernBERT gpt2 zero-shot \
    --sampling-methods random ssepy minority info_gain_lightgbm accuracy_lightgbm isolation \
    --sample-sizes 10 100 1000 \
    --max-runs 5
```

**Parameters:**
- `base_path` (required): Base experiment directory path
- `--imbalances` (required): Imbalance levels (e.g., '01', '05')
- `--models` (optional): Models to analyse (default: distilbert, ModernBERT, gpt2, zero-shot)
- `--sampling-methods` (optional): Sampling methods to analyse (default: random, ssepy, minority, info_gain_lightgbm, accuracy_lightgbm, isolation)
- `--sample-sizes` (optional): Specific sample sizes to analyse (e.g., 10 100 1000)
- `--max-runs` (optional): Maximum number of runs per sampling method (None = use all available)

**Description:**
- Class imbalance analysis across different sample sizes
- Sampling strategy bias assessment with statistical summaries
- Distribution visualization and percentage tables
- Outputs summary CSV files and LaTeX tables to `{base_path}/tables/class_distributions/`
- Creates visualizations in `{base_path}/figures/class_distributions/`

### `aggregate_replay.py`
Aggregates replay experiments across multiple models and imbalance levels. This script has hardcoded configuration and analyses label transfer performance.

**Usage:**
```bash
python scripts/analysis/aggregate_replay.py
```

**Parameters:**
- No command-line arguments (uses hardcoded configuration)
- Hardcoded paths: `outputs/reddit_dataset_12/one-vs-all/football/42_05`
- Hardcoded models: distilbert, gpt2, ModernBERT, zero-shot
- Hardcoded imbalances: 05, 01, 001
- Hardcoded samplers: accuracy_lightgbm, info_gain_lightgbm, minority, ssepy

**Description:**
- Replay experiment aggregation with bootstrap RMSE analysis
- Cross-model strategy consistency analysis
- Performance stability assessment across configurations
- Generates tables and figures in `outputs/.../tables/aggregate_replay/` and `outputs/.../figures/aggregate_replay/`

### `analyse_replay.py`
Analyses individual replay transfer results for a specific source model.

**Usage:**
```bash
python scripts/analysis/analyse_replay.py \
    /path/to/source/model/results/directory
```

**Parameters:**
- `source_model_dir` (required): Path to the source model's results directory

**Description:**
- Individual replay experiment analysis
- Generates plots comparing replay results across different target configurations
- Uses predefined configs: distilbert_default, gpt2_default, ModernBERT_short, zero-shot_default

## Data Requirements

These scripts expect data in the following structure:
```
outputs/
├── reddit_dataset_12/
│   └── one-vs-all/
│       └── football/
│           └── 42_05/
│               ├── distilbert/
│               │   └── default/
│               │       └── eval_outputs/
│               │           ├── 05/
│               │           ├── 01/
│               │           └── 001/
│               │               ├── random/
│               │               ├── accuracy_lightgbm/
│               │               └── ...
│               ├── gpt2/
│               └── ModernBERT/
```

Each strategy directory should contain:
- `stats_full.json`: Complete statistics for all samples
- `sample_*.json`: Sample selection files for different seeds
- `metrics_*.json`: Computed metrics for different sample sizes
