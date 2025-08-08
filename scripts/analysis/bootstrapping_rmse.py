import argparse

from arc_tigers.eval.aggregate_utils import (
    collect_se_values,
    generate_tables,
    get_evaluate_steps,
    perform_bootstrap,
    print_results,
    save_json_results,
    stack_se_values,
)


def main(
    base_path: str,
    models: list[str],
    imbalances: list[str],
    sampling_methods: list[str],
    verbose: bool = False,
) -> None:
    """Main function to run bootstrap RMSE analysis."""
    metrics = [
        "accuracy",
        "loss",
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

    # Collect SE values
    se_vals = collect_se_values(
        base_path, models, imbalances, sampling_methods, metrics
    )

    # Stack SE values for bootstrap analysis
    stacked_se_vals = stack_se_values(se_vals, imbalances, sampling_methods, metrics)

    # Perform bootstrap analysis
    boot_results = perform_bootstrap(
        stacked_se_vals, imbalances, sampling_methods, metrics
    )
    if verbose:
        # Print results
        print_results(
            boot_results,
            stacked_se_vals,
            imbalances,
            sampling_methods,
            metrics,
        )

    # Generate tables
    generate_tables(
        boot_results,
        stacked_se_vals,
        evaluate_steps,
        imbalances,
        sampling_methods,
        metrics,
        verbose,
    )

    # Save JSON results
    save_json_results(boot_results, stacked_se_vals)


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
        "-m",
        nargs="+",
        default=["distilbert", "ModernBERT", "gpt2", "zero-shot"],
        help="Model names to include",
    )
    parser.add_argument(
        "--imbalances",
        "-i",
        nargs="+",
        default=["05", "01", "001"],
        help="Test imbalance levels (e.g. '05', '01')",
    )
    parser.add_argument(
        "--sampling-methods",
        "-s",
        nargs="+",
        default=[
            "random",
            "ssepy",
            "minority",
            "info_gain_lightgbm",
            "accuracy_lightgbm",
        ],
        help="Sampling strategies to evaluate",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed output and tables",
    )

    args = parser.parse_args()
    main(
        args.base_path,
        args.models,
        args.imbalances,
        args.sampling_methods,
        args.verbose,
    )
