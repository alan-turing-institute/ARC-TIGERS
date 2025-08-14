import argparse
from pathlib import Path

from arc_tigers.eval.aggregate_utils import compute_class_distributions
from arc_tigers.eval.plotting import (
    class_pct_summary_by_size,
    class_percentage_table,
    plot_class_distribution_by_size,
    save_latex_summary_table,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze class distribution vs sample size from metrics files"
    )
    parser.add_argument("base_path", help="Base experiment path")
    parser.add_argument(
        "--imbalances",
        nargs="+",
        default=["01"],
        help="Imbalance levels (e.g., '01', '05')",
        required=True,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilbert", "ModernBERT", "gpt2"],
        help="Models to analyze",
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
        ],
        help="Sampling methods to analyze",
    )
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        help="Specific sample sizes to analyze (e.g., --sample-sizes 10 100 1000)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Maximum number of runs per sampling method (None = use all available)",
    )

    args = parser.parse_args()

    for imbalance in args.imbalances:
        # Analyze sampling strategies from metrics files
        results_df = compute_class_distributions(
            args.base_path,
            args.imbalance,
            args.models,
            args.sampling_methods,
            args.sample_sizes,
            args.max_runs,
        )

        if results_df.empty:
            print("No data found. Please check your paths and parameters.")
            return

        # Create summary table
        summary_df = class_pct_summary_by_size(results_df)
        if not summary_df.empty:
            print(
                f"\nSummary statistics calculated for {len(summary_df)}"
                " method-size combinations"
            )

        # Create output directory
        table_dir = Path(args.base_path) / "tables" / "class_distributions" / imbalance
        plots_dir = Path(args.base_path) / "figures" / "class_distributions" / imbalance

        table_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save summary results
        if not summary_df.empty:
            summary_file = table_dir / "summary_class_distribution_from_metrics.csv"
            summary_df.to_csv(summary_file, index=False)

            # Create and save clean percentage table (like aggregate analysis scripts)
            percentage_table = class_percentage_table(summary_df)
            if not percentage_table.empty:
                # Save as CSV
                percentage_file = table_dir / "class_percentage_by_sample_size.csv"
                percentage_table.to_csv(percentage_file)

                # Save as LaTeX
                latex_percentage_file = (
                    table_dir / f"class_percentage_by_sample_size_{imbalance}.tex"
                )
                caption = (
                    f"Positive class percentage by sampling method and sample size "
                    f"(imbalance {imbalance})."
                    f"Values show mean percentage across runs."
                )
                save_latex_summary_table(
                    percentage_table, str(latex_percentage_file), caption
                )

                print("\nClass Percentage Summary Table:")
                print(percentage_table)

        # Create and save plots
        plot_class_distribution_by_size(results_df, str(plots_dir), imbalance)

    results_df.to_csv(Path(args.base_path) / "full_results.csv", index=False)


if __name__ == "__main__":
    main()
