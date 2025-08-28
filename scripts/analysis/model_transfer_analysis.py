import os

import numpy as np
import pandas as pd
from scipy.stats import bootstrap

from arc_tigers.eval.utils import get_se_values

# Configuration
base_dir = "outputs/reddit_dataset_12/one-vs-all/football/42_05"
tables_dir = f"{base_dir}/tables/model_transfer_analysis"
figures_dir = f"{base_dir}/figures/model_transfer_analysis"
data_dir = f"{tables_dir}/raw_data"

# Create output directories if they don't exist
os.makedirs(tables_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

SAMPLING_STRATEGIES = ["accuracy_lightgbm", "info_gain_lightgbm", "minority", "ssepy"]
models = {
    "DistilBERT": ("distilbert", "default"),
    "GPT-2": ("gpt2", "default"),
    "ModernBERT": ("ModernBERT", "short"),
    "Zero-shot": ("zero-shot", "default"),
}
imbalances = ["05", "01", "001"]
METRICS = [
    "accuracy",
    "average_precision",
    "f1_0",
    "f1_1",
    "precision_0",
    "precision_1",
    "recall_0",
    "recall_1",
]


def nan_root_mean(values):
    return np.sqrt(np.nanmean(values))


def get_aggregate_rmse(replays_dir):
    """Get the aggregate RMSE for a specific model transfer scenario."""
    bootstrap_se_vals = []
    for sample_strategy in SAMPLING_STRATEGIES:
        replay_metrics_paths = f"{replays_dir}/{sample_strategy}/all_seeds/"
        se_values = get_se_values(replay_metrics_paths)
        for metric in METRICS:
            bootstrap_se_vals.append(se_values[metric].flatten())

    bootstrap_se_vals = np.concatenate(bootstrap_se_vals)

    bootstrap_results = bootstrap(
        (bootstrap_se_vals,),
        nan_root_mean,
        confidence_level=0.95,
        n_resamples=1000,
    )
    return (
        nan_root_mean(bootstrap_se_vals),
        (
            bootstrap_results.confidence_interval.low,
            bootstrap_results.confidence_interval.high,
        ),
    )


def model_transfer_matrix():
    """Create a transfer matrix for model performance across different imbalances."""

    # Initialize list to collect results
    results = []

    for imbalance in imbalances:
        for model1_name, model1 in models.items():
            for model2_name, model2 in models.items():
                if model1 == model2:
                    continue
                # Simulate getting model performance for each imbalance
                replay_dir = (
                    f"{base_dir}/replays/{imbalance}/{model1[0]}_{model1[1]}"
                    f"/from_{model2[0]}_{model2[1]}"
                )
                aggregate_rmse = get_aggregate_rmse(
                    replay_dir,
                )

                # Add to results
                results.append(
                    {
                        "imbalance": imbalance,
                        "source_model": model1_name,
                        "target_model": model2_name,
                        "rmse_mean": aggregate_rmse[0],
                        "rmse_ci_low": aggregate_rmse[1][0],
                        "rmse_ci_high": aggregate_rmse[1][1],
                    }
                )

                print(model1, model2, aggregate_rmse)

    # Create DataFrame from results
    return pd.DataFrame(results)


def create_transfer_matrices(df):
    """
    Create 3 4x4 matrices for each imbalance level showing model transfer performance.
    """

    # Get model names for indexing
    model_names = list(models.keys())

    matrices = {}

    for imbalance in imbalances:
        # Filter data for this imbalance level
        df_imb = df[df["imbalance"] == imbalance]

        # Initialize matrices for mean, CI low, and CI high
        mean_matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=float)
        ci_low_matrix = pd.DataFrame(
            index=model_names, columns=model_names, dtype=float
        )
        ci_high_matrix = pd.DataFrame(
            index=model_names, columns=model_names, dtype=float
        )

        # Fill the matrices
        for _, row in df_imb.iterrows():
            source = row["source_model"]
            target = row["target_model"]

            # Source model is the row (first column)
            # target model is the column (first row)
            mean_matrix.loc[source, target] = row["rmse_mean"]
            ci_low_matrix.loc[source, target] = row["rmse_ci_low"]
            ci_high_matrix.loc[source, target] = row["rmse_ci_high"]

        # Set diagonal to NaN (same model transfers)
        for model in model_names:
            mean_matrix.loc[model, model] = np.nan
            ci_low_matrix.loc[model, model] = np.nan
            ci_high_matrix.loc[model, model] = np.nan

        # Store matrices
        matrices[imbalance] = {
            "mean": mean_matrix,
            "ci_low": ci_low_matrix,
            "ci_high": ci_high_matrix,
        }

        # Print matrices for this imbalance level
        print(f"\n{'=' * 60}")
        print(f"IMBALANCE LEVEL: {imbalance}")
        print(f"{'=' * 60}")

        print("\nMean RMSE Matrix:")
        print(mean_matrix.round(4))

        print("\nConfidence Interval Lower Bound:")
        print(ci_low_matrix.round(4))

        print("\nConfidence Interval Upper Bound:")
        print(ci_high_matrix.round(4))

        # Save matrices to CSV
        mean_matrix.to_csv(f"{tables_dir}/transfer_matrix_mean_{imbalance}.csv")
        ci_low_matrix.to_csv(f"{tables_dir}/transfer_matrix_ci_low_{imbalance}.csv")
        ci_high_matrix.to_csv(f"{tables_dir}/transfer_matrix_ci_high_{imbalance}.csv")

        print(f"\nMatrices saved to {tables_dir}/transfer_matrix_*_{imbalance}.csv")

    return matrices


def create_combined_matrix(matrices):
    """Create combined matrices showing mean ± (CI_low, CI_high) for each imbalance."""

    model_names = list(models.keys())

    for imbalance in imbalances:
        # Create combined string matrix
        model_names_table_map = {name: f"\\textbf{{{name}}}" for name in model_names}
        combined_matrix = pd.DataFrame(
            index=model_names_table_map.values(),
            columns=model_names_table_map.values(),
            dtype=object,
        )

        mean_mat = matrices[imbalance]["mean"]
        ci_low_mat = matrices[imbalance]["ci_low"]
        ci_high_mat = matrices[imbalance]["ci_high"]

        for source in model_names:
            for target in model_names:
                if source == target:
                    combined_matrix.loc[
                        model_names_table_map[source], model_names_table_map[target]
                    ] = "—"
                else:
                    mean_val = mean_mat.loc[source, target]
                    ci_low_val = ci_low_mat.loc[source, target]
                    ci_high_val = ci_high_mat.loc[source, target]

                    if pd.notna(mean_val):
                        combined_matrix.loc[
                            model_names_table_map[source], model_names_table_map[target]
                        ] = (
                            f"${mean_val:.4f}"
                            f"_{{{ci_low_val:.4f}}}"
                            f"^{{{ci_high_val:.4f}}}$"
                        )
                    else:
                        combined_matrix.loc[
                            model_names_table_map[source], model_names_table_map[target]
                        ] = "N/A"

        # Create the LaTeX table with custom formatting
        imbalance_percentage = int(float(imbalance[0] + "." + imbalance[1:]) * 100)
        latex_content = combined_matrix.to_latex(
            label=f"tab:model_transfer_matrix_{imbalance}",
            caption="Model transfer matrix showing RMSE values when transferring "
            f"between models. Results are shown for an imbalance"
            f" of {imbalance_percentage}\\%.",
            escape=False,
            column_format="l" + "c" * (len(model_names)),
        )

        # Replace the empty top-left cell with "Source / Target Model"
        latex_content = latex_content.replace(
            "\\toprule\n & ",
            "\\toprule\n & \\multicolumn{4}{c}{\\textbf{Target Model}} \\\\"
            "\n\\textbf{Source Model} & ",
        )

        latex_content = latex_content.replace(
            "\\begin{table}\n", "\\begin{table}\n\\centering\n"
        )

        # Write the modified LaTeX content to file
        with open(f"{tables_dir}/transfer_matrix_combined_{imbalance}.tex", "w") as f:
            f.write(latex_content)


if __name__ == "__main__":
    df_results = model_transfer_matrix()
    matrices = create_transfer_matrices(df_results)
    # Create combined matrices with mean ± bounds
    create_combined_matrix(matrices)
