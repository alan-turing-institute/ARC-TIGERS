import json
import os
from glob import glob

import numpy as np
import pandas as pd
import yaml
from scipy.stats import bootstrap

from arc_tigers.eval.utils import get_metric_stats, get_se_values


def sqrt_mean(x: np.ndarray, axis: int | None = None):
    """Return sqrt of the mean along axis (no forced scalar cast).
    When axis is None a scalar is returned; otherwise an array result is fine
    for scipy.bootstrap which supplies axis=-1 and expects reduction.
    """
    return np.sqrt(np.nanmean(x, axis=axis))


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


def stack_se_values(
    se_vals: dict,
    imbalances: list[str],
    sampling_methods: list[str],
    metrics: list[str],
) -> dict:
    """Stack SE values into numpy arrays for bootstrap analysis."""
    return {
        imbalance: {
            strat: {
                metric: np.squeeze(
                    np.hstack(list(se_vals[imbalance][strat][metric].values())), axis=0
                )
                for metric in metrics
            }
            for strat in sampling_methods
        }
        for imbalance in imbalances
    }


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
                n_repeats = stacked_se_vals[imbalance][sampling_strat][metric].shape[1]
                for n in range(n_repeats):
                    vals = stacked_se_vals[imbalance][sampling_strat][metric][:, n]
                    boot_results[imbalance][sampling_strat][metric][n] = bootstrap(
                        (vals,),
                        sqrt_mean,
                        n_resamples=1000,
                        method="basic",
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
    verbose: bool = False,
) -> None:
    """Generate LaTeX tables for bootstrap results."""
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

                    # Use mean across models if multiple
                    values.append(np.mean(step_stats))

                df.loc[len(df)] = [strategy, *values]

            # Save LaTeX table
            save_dir = f"tmp/tables/bootstrap_rmse/{imbalance}/"
            os.makedirs(save_dir, exist_ok=True)
            latex_table = df.to_latex(index=False, float_format="%.4f")
            with open(f"{save_dir}/{metric}.tex", "w") as f:
                f.write(latex_table)
            if verbose:
                print(f"Bootstrap RMSE table saved to {save_dir}/{metric}.tex")
                print(df)


def save_json_results(boot_results: dict, stacked_se_vals: dict) -> None:
    """Save bootstrap results to JSON format."""
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/bootstrapped_rmse_results.json", "w") as f:
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
