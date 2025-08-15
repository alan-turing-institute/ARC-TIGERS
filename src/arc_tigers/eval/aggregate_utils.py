import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import bootstrap

from arc_tigers.eval.plotting import SAMPLE_STRAT_NAME_MAP
from arc_tigers.eval.utils import (
    get_class_percentages,
    get_metric_stats,
    get_se_differences,
    get_se_values,
    load_metrics_file_to_df,
)


def sqrt_mean(x: np.ndarray, axis: int | None = None):
    """
    Return sqrt of the mean along axis (no forced scalar cast).
    When axis is None a scalar is returned; otherwise an array result is fine
    for scipy.bootstrap which supplies axis=-1 and expects reduction.
    The sqrt of the mean is taken because the square error is passed as x.

    Args:
        x: Input array of square errors.
        axis: Axis along which to compute the mean. If None, computes over the entire
        array
    Returns:
        The square root of the mean of the input array along the specified axis.
    """
    mean = np.nanmean(x, axis=axis)
    negative_mask = np.where(mean < 0, -1, 1)  # Convert boolean to -1/1
    return np.sqrt(abs(mean)) * negative_mask


def get_evaluate_steps(
    base_path: str,
    models: list[str],
    imbalances: list[str],
    sampling_methods: list[str],
) -> list[int]:
    """Extract evaluate_steps from the first available config file."""
    evaluate_steps = None
    for imbalance in imbalances:
        for model in models:
            for sampling_method in sampling_methods:
                path_pattern = os.path.join(
                    base_path,
                    model,
                    "*",
                    "eval_outputs",
                    imbalance,
                    sampling_method,
                )
                config_dirs = glob(path_pattern)
                if config_dirs:
                    config_path = os.path.join(config_dirs[0], "config.yaml")
                    if os.path.exists(config_path):
                        with open(config_path) as f:
                            config = yaml.safe_load(f)
                            evaluate_steps = config.get("evaluate_steps", [])
                            break
            if evaluate_steps:
                break
        if evaluate_steps:
            break

    if not evaluate_steps:
        msg = "Could not find evaluate_steps in any config file"
        raise ValueError(msg)

    return evaluate_steps


def collect_se_values(
    base_path: str,
    models: list[str],
    imbalances: list[str],
    sampling_methods: list[str],
    metrics: list[str],
) -> dict:
    """Collect standard error values from evaluation outputs."""
    se_vals: dict[str, dict[str, dict[str, dict[str, list]]]] = {
        imbalance: {
            strat: {metric: {model: [] for model in models} for metric in metrics}
            for strat in sampling_methods
        }
        for imbalance in imbalances
    }

    for imbalance in imbalances:
        for model in models:
            for sampling_method in sampling_methods:
                path_pattern = os.path.join(
                    base_path,
                    model,
                    "*",
                    "eval_outputs",
                    imbalance,
                    sampling_method,
                )

                config_dirs = glob(path_pattern)
                if not config_dirs:
                    continue
                first_config = config_dirs[0]

                run_se_vals = get_se_values(first_config)

                for metric in metrics:
                    se_vals[imbalance][sampling_method][metric][model].append(
                        run_se_vals[metric]
                    )

    return se_vals


def collect_se_differences(
    base_path: str,
    models: list[str],
    imbalances: list[str],
    sampling_methods: list[str],
    metrics: list[str],
) -> dict:
    """Collect standard error values from evaluation outputs."""
    se_diff_vals: dict[str, dict[str, dict[str, dict[str, list]]]] = {
        imbalance: {
            strat: {metric: {model: [] for model in models} for metric in metrics}
            for strat in sampling_methods
        }
        for imbalance in imbalances
    }

    for imbalance in imbalances:
        for model in models:
            for sampling_method in sampling_methods:
                path_pattern = os.path.join(
                    base_path,
                    model,
                    "*",
                    "eval_outputs",
                    imbalance,
                    sampling_method,
                )
                random_pattern = os.path.join(
                    base_path,
                    model,
                    "*",
                    "eval_outputs",
                    imbalance,
                    "random",
                )

                target_config_dirs = glob(path_pattern)
                if not target_config_dirs:
                    err_msg = f"Warning: No configs found for model '{model}' "
                    raise ValueError(err_msg)
                random_config_dirs = glob(random_pattern)
                if not random_config_dirs:
                    err_msg = f"Warning: No random baseline found for model '{model}' "
                    raise ValueError(err_msg)

                first_target_config = target_config_dirs[0]
                first_random_config = random_config_dirs[0]

                run_se_diffs = get_se_differences(
                    first_target_config, first_random_config
                )

                for metric in metrics:
                    se_diff_vals[imbalance][sampling_method][metric][model].append(
                        run_se_diffs[metric]
                    )

    return se_diff_vals


def stack_values(
    se_vals: dict,
    imbalances: list[str],
    sampling_methods: list[str],
    metrics: list[str],
    step_indices: list[int] | None = None,
) -> dict:
    """Stack SE values into numpy arrays for bootstrap analysis."""
    stacked: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for imbalance in imbalances:
        stacked[imbalance] = {}
        for strat in sampling_methods:
            stacked[imbalance][strat] = {}
            for metric in metrics:
                # Stack all model values for this metric
                all_values = list(se_vals[imbalance][strat][metric].values())
                stacked_array = np.squeeze(np.hstack(all_values), axis=0)

                # Filter to selected step indices if provided
                if step_indices is not None:
                    # Handle both 1D and 2D arrays
                    if stacked_array.ndim == 1:
                        stacked_array = stacked_array[step_indices]
                    else:
                        stacked_array = stacked_array[:, step_indices]

                stacked[imbalance][strat][metric] = stacked_array

    return stacked


def perform_bootstrap(
    stacked_se_vals: dict,
    imbalances: list[str],
    sampling_methods: list[str],
    metrics: list[str],
) -> dict:
    """Perform bootstrap analysis on stacked SE values."""
    boot_results: dict[str, dict[str, dict[str, dict[int, object]]]] = {
        imbalance: {
            strat: {metric: {} for metric in metrics} for strat in sampling_methods
        }
        for imbalance in imbalances
    }

    for imbalance in imbalances:
        for sampling_strat in sampling_methods:
            for metric in metrics:
                sampling_counts = stacked_se_vals[imbalance][sampling_strat][
                    metric
                ].shape[1]
                for n in range(sampling_counts):
                    vals = stacked_se_vals[imbalance][sampling_strat][metric][:, n]
                    boot_results[imbalance][sampling_strat][metric][n] = bootstrap(
                        (vals,),
                        sqrt_mean,
                        n_resamples=1000,
                    )

    return boot_results


def print_results(
    boot_results: dict,
    stacked_se_vals: dict,
    imbalances: list[str],
    sampling_methods: list[str],
    metrics: list[str],
) -> None:
    """Print bootstrap results."""
    print("Bootstrapped RMSE results:")
    for imbalance in imbalances:
        for sampling_strat in sampling_methods:
            for metric in metrics:
                for n, result in boot_results[imbalance][sampling_strat][
                    metric
                ].items():
                    # Handle older SciPy versions without .statistic attribute
                    vals = stacked_se_vals[imbalance][sampling_strat][metric][:, n]
                    pre_stat = sqrt_mean(vals)
                    stat = getattr(result, "statistic", pre_stat)
                    # Normalise to scalar
                    stat_val = float(np.asarray(stat).ravel()[0])

                    ci_obj = getattr(result, "confidence_interval", None)
                    if ci_obj is not None:
                        ci_low = float(ci_obj.low)
                        ci_high = float(ci_obj.high)
                    else:
                        ci_low = np.nan
                        ci_high = np.nan

                    print(
                        f"Imbalance: {imbalance}, Sampling: {sampling_strat}, "
                        f"Metric: {metric}, N: {n}, Value: {stat_val:.4f}, "
                        f"CI: ({ci_low:.4f}, {ci_high:.4f})"
                    )


def generate_tables(
    boot_results: dict,
    stacked_se_vals: dict,
    evaluate_steps: list[int],
    imbalances: list[str],
    sampling_methods: list[str],
    metrics: list[str],
    save_dir: str,
    verbose: bool = False,
) -> None:
    """Generate LaTeX tables for bootstrap results."""

    metric_aggregate_results: dict[str, list] = {metric: [] for metric in metrics}
    for imbalance in imbalances:
        for metric in metrics:
            # Create DataFrame for this imbalance/metric combination
            df = pd.DataFrame(columns=["strategy", *evaluate_steps])

            for strategy in sampling_methods:
                values = []
                for step_idx, _ in enumerate(evaluate_steps):
                    # Collect bootstrap statistics for this sample size
                    step_stats = []
                    if (
                        strategy in boot_results[imbalance]
                        and metric in boot_results[imbalance][strategy]
                        and step_idx in boot_results[imbalance][strategy][metric]
                    ):
                        result = boot_results[imbalance][strategy][metric][step_idx]
                        # Handle older SciPy versions
                        vals = stacked_se_vals[imbalance][strategy][metric][:, step_idx]
                        pre_stat = sqrt_mean(vals)
                        stat = getattr(result, "statistic", pre_stat)
                        stat_val = float(np.asarray(stat).ravel()[0])
                        step_stats.append(stat_val)

                    # Use mean across models if multiple, handle empty arrays
                    if step_stats:
                        values.append(np.mean(step_stats))
                    else:
                        values.append(np.nan)

                df.loc[len(df)] = [strategy, *values]

            # Save LaTeX table
            table_dir = f"{save_dir}/{imbalance}/"
            os.makedirs(table_dir, exist_ok=True)
            latex_table = df.to_latex(index=False, float_format="%.4f")
            with open(f"{table_dir}/{metric}.tex", "w") as f:
                f.write(latex_table)
            if verbose:
                print(f"Bootstrap RMSE table saved to {table_dir}/{metric}.tex")
                print(df)

            metric_aggregate_results[metric].append(df.to_numpy()[:, 1:])

    # Save all results as a single DataFrame
    agg_table_dir = f"{save_dir}/aggregate/"
    os.makedirs(agg_table_dir, exist_ok=True)
    sampling_methods_labels = [
        SAMPLE_STRAT_NAME_MAP.get(method, method) for method in sampling_methods
    ]
    for metric, results in metric_aggregate_results.items():
        # take mean across all tables
        # Round to 3 significant figures
        stacked_df = np.dstack(results)
        mean_over_imbalances = np.mean(stacked_df, axis=-1)
        mean_over_sample_counts = np.mean(stacked_df, axis=1)
        mean_df = pd.DataFrame(
            mean_over_imbalances, columns=evaluate_steps, index=sampling_methods_labels
        )
        mean_df.to_latex(
            f"{agg_table_dir}/{metric}_over_imbalances.tex",
            index=True,
            float_format="%.4f",
            caption=f"Bootstrapped RMSE for "
            f"{metric.replace('_', ' ')} across sampling methods.",
            label=f"tab:bootstrapped_rmse_{metric}",
        )

        mean_df = pd.DataFrame(
            mean_over_sample_counts,
            columns=imbalances,
            index=sampling_methods_labels,
        )
        mean_df.to_latex(
            f"{agg_table_dir}/{metric}_over_sample_count.tex",
            index=True,
            float_format="%.4f",
            caption=f"Bootstrapped RMSE for "
            f"{metric.replace('_', ' ')} across sampling methods.",
            label=f"tab:bootstrapped_rmse_{metric}",
        )


def save_json_results(boot_results: dict, stacked_se_vals: dict, save_dir: str) -> None:
    """Save bootstrap results to JSON format."""
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/all_results.json", "w") as f:
        # Convert BootstrapResult objects to serializable format
        serializable_results: dict[
            str, dict[str, dict[str, dict[int, dict[str, float]]]]
        ] = {}
        for imb, imb_results in boot_results.items():
            serializable_results[imb] = {}
            for strat, strat_results in imb_results.items():
                serializable_results[imb][strat] = {}
                for metric, metric_results in strat_results.items():
                    serializable_results[imb][strat][metric] = {}
                    for n, result in metric_results.items():
                        # Handle older SciPy versions
                        vals = stacked_se_vals[imb][strat][metric][:, n]
                        pre_stat = sqrt_mean(vals)
                        stat = getattr(result, "statistic", pre_stat)
                        stat_val = float(np.asarray(stat).ravel()[0])

                        ci_obj = getattr(result, "confidence_interval", None)
                        if ci_obj is not None:
                            ci_low = float(ci_obj.low)
                            ci_high = float(ci_obj.high)
                        else:
                            ci_low = float("nan")
                            ci_high = float("nan")

                        serializable_results[imb][strat][metric][n] = {
                            "statistic": stat_val,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        }
        json.dump(serializable_results, f, indent=2)


def better_than_random(
    base_path: str,
    model: str,
    test_imbalance: str,
    sampling_method: str,
    metric: str,
) -> dict[int, float]:
    """
    Compute percent improvement of a sampling method
    over random baseline for a specific metric.

    Args:
        base_path: Base experiment path (e.g. 'outputs/…/42_05').
        model: Model name directory.
        test_imbalance: Test imbalance identifier (e.g. '05').
        sampling_method: Strategy to evaluate (e.g. 'info_gain').
        metric: Metric name (e.g. 'accuracy', 'f1_1', 'loss').

    Returns:
        Mapping from number of labels to percent
        improvement over random (positive means better).
    """
    # locate random baseline path
    rand_pattern = os.path.join(
        base_path, model, "*", "eval_outputs", test_imbalance, "random"
    )
    rand_matches = sorted(glob(rand_pattern))
    if not rand_matches:
        msg = f"No random baseline found for model={model}, imbalance={test_imbalance}"
        raise ValueError(msg)
    rand_dir = rand_matches[0]
    rand_stats, _ = get_metric_stats(rand_dir, plot=False)

    # locate strategy path
    strat_pattern = os.path.join(
        base_path, model, "*", "eval_outputs", test_imbalance, sampling_method
    )
    strat_matches = sorted(glob(strat_pattern))
    if not strat_matches:
        msg = f"No path for sampling method '{sampling_method}' for model={model}"
        raise ValueError(msg)
    strat_dir = strat_matches[0]
    strat_stats, _ = get_metric_stats(strat_dir, plot=False)

    # compare values across all sample counts
    n_labels = rand_stats["n_labels"]
    results: dict[int, float] = {}
    for idx, n in enumerate(n_labels):
        rv = rand_stats[metric]["mse"][idx]
        sv = strat_stats[metric]["mse"][idx]
        imp = ((sv - rv) / rv * 100) if rv != 0 else 0.0
        results[int(n)] = imp
    return results


def compute_class_distributions(
    base_path: str,
    imbalance: str,
    models: list[str],
    sampling_methods: list[str],
    sample_sizes: list[int],
    max_runs_per_method: int | None = None,
) -> pd.DataFrame:
    """Analyze class distribution from metrics files."""

    results = []

    for model in models:
        for sampling_method in sampling_methods:
            # Find metrics files for this configuration
            pattern = os.path.join(
                base_path,
                model,
                "*",
                "eval_outputs",
                imbalance,
                sampling_method,
                "metrics_*.csv",
            )
            metrics_files = glob(pattern)

            if not metrics_files:
                print(f"Warning: No metrics files found for {model}/{sampling_method}")
                continue

            # Limit number of runs for faster processing (set to None for all files)
            if max_runs_per_method is not None:
                metrics_files = metrics_files[:max_runs_per_method]

            # Analyze each metrics file
            for metrics_file in metrics_files:
                df = load_metrics_file_to_df(metrics_file)
                if df.empty:
                    continue

                # Calculate class percentages
                df = get_class_percentages(df)
                if df.empty:
                    continue

                # Filter by requested sample sizes if specified
                if sample_sizes is not None:
                    df = df[df["sample_size"].isin(sample_sizes)]
                    if df.empty:
                        continue

                # Extract run ID from filename
                run_id = Path(metrics_file).stem.replace("metrics_", "")

                # Add metadata to each row
                df["model"] = model
                df["sampling_method"] = sampling_method
                df["run_id"] = run_id

                results.append(df)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
