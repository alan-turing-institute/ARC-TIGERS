import argparse

from arc_tigers.eval.aggregate_utils import (
    all_metrics_table,
    collect_se_differences,
    collect_se_values,
    generate_tables,
    get_evaluate_steps,
    group_metrics,
    grouped_bootstrap,
    perform_bootstrap,
    print_results,
    save_json_results,
    stack_values,
)
from arc_tigers.eval.plotting import generate_grouped_tables


def main(
    base_path: str,
    models: list[str],
    imbalances: list[str],
    sampling_methods: list[str],
    verbose: bool = False,
    selected_steps: list[int] | None = None,
) -> None:
    """Main function to run bootstrap RMSE analysis."""
    metrics = [
        "accuracy",
        "average_precision",
        "f1_0",
        "f1_1",
        "precision_0",
        "precision_1",
        "recall_0",
        "recall_1",
    ]

    # Get evaluate_steps from first available config
    evaluate_steps = get_evaluate_steps(base_path, models, imbalances, sampling_methods)
    if verbose:
        print(f"Found evaluate_steps: {evaluate_steps}")

    # Filter to selected steps if provided
    if selected_steps is not None:
        # Find indices of selected steps in evaluate_steps
        step_indices = []
        filtered_steps = []
        for step in selected_steps:
            if step in evaluate_steps:
                idx = evaluate_steps.index(step)
                step_indices.append(idx)
                filtered_steps.append(step)
            else:
                msg = (
                    f"Warning: Selected step {step} not found in "
                    f"available steps {evaluate_steps}"
                )
                print(msg)

        if not filtered_steps:
            error_msg = (
                f"None of the selected steps {selected_steps} are "
                f"available in {evaluate_steps}"
            )
            raise ValueError(error_msg)

        evaluate_steps = filtered_steps
        if verbose:
            print(f"Filtered to selected steps: {evaluate_steps}")
    else:
        step_indices = None

    # Collect SE values
    se_vals = collect_se_values(
        base_path, models, imbalances, sampling_methods, metrics
    )
    # Remove "random" from sampling_methods if present
    diff_sampling_methods = [
        method for method in sampling_methods if method != "random"
    ]

    se_diffs = collect_se_differences(
        base_path, models, imbalances, diff_sampling_methods, metrics
    )

    # Stack SE values for bootstrap analysis
    stacked_se_vals = stack_values(
        se_vals, imbalances, sampling_methods, metrics, step_indices
    )
    # Stack SE differences for bootstrap analysis
    stacked_se_diffs = stack_values(
        se_diffs, imbalances, diff_sampling_methods, metrics, step_indices
    )

    # Perform bootstrap analysis
    raw_boot_results = perform_bootstrap(
        stacked_se_vals, imbalances, sampling_methods, metrics
    )
    diff_boot_results = perform_bootstrap(
        stacked_se_diffs, imbalances, diff_sampling_methods, metrics
    )

    grouped_metrics = group_metrics(stacked_se_vals, imbalances, sampling_methods)
    grouped_bootstrap_results = grouped_bootstrap(
        grouped_metrics, imbalances, sampling_methods
    )

    all_metrics_table(
        se_vals,
        imbalances,
        sampling_methods,
        metrics,
        save_dir=f"{base_path}/tables/bootstrap_rmse/all_metrics/",
    )

    if verbose:
        # Print results
        print_results(
            raw_boot_results,
            stacked_se_vals,
            imbalances,
            sampling_methods,
            metrics,
        )

    # Generate tables
    raw_dir = f"{base_path}/tables/bootstrap_rmse/raw"
    generate_tables(
        raw_boot_results,
        stacked_se_vals,
        evaluate_steps,
        imbalances,
        sampling_methods,
        metrics,
        raw_dir,
        verbose,
    )
    diff_dir = f"{base_path}/tables/bootstrap_rmse/differences"
    generate_tables(
        diff_boot_results,
        stacked_se_diffs,
        evaluate_steps,
        imbalances,
        diff_sampling_methods,
        metrics,
        diff_dir,
        verbose,
    )

    # Save JSON results
    save_json_results(raw_boot_results, stacked_se_vals, raw_dir)
    save_json_results(diff_boot_results, stacked_se_diffs, diff_dir)

    # Generate grouped bootstrap tables for each imbalance level
    grouped_dir = f"{base_path}/tables/bootstrap_rmse/grouped"
    generate_grouped_tables(
        grouped_bootstrap_results,
        evaluate_steps,
        imbalances,
        sampling_methods,
        grouped_dir,
        verbose,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap RMSE analysis of sampling strategies"
    )
    parser.add_argument(
        "base_path",
        help="Base experiment path",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilbert", "ModernBERT", "gpt2", "zero-shot"],
        help="Model names to include",
    )
    parser.add_argument(
        "--imbalances",
        nargs="+",
        default=["05", "01", "001"],
        help="Test imbalance levels (e.g. '05', '01')",
    )
    parser.add_argument(
        "--sampling-methods",
        nargs="+",
        default=[
            "random",
            "ssepy",
            "minority",
            "info_gain_lightgbm",
            "accuracy_lightgbm",
            "isolation",
        ],
        help="Sampling strategies to evaluate",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output and tables",
    )
    parser.add_argument(
        "--selected-steps",
        nargs="+",
        type=int,
        help="Only include specific evaluation steps (e.g., 10 100 500)",
    )

    args = parser.parse_args()
    main(
        args.base_path,
        args.models,
        args.imbalances,
        args.sampling_methods,
        args.verbose,
        args.selected_steps,
    )
