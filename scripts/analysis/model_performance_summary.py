import argparse
import json
import os
from glob import glob

import pandas as pd

from arc_tigers.eval.plotting import MODEL_NAME_MAP

MODERN_BERT_MAP = {"fp16": "small", "200k": "large", "short": "large"}


def main():
    parser = argparse.ArgumentParser(
        description="Model performance summary across training runs"
    )
    parser.add_argument("base_path", help="Base experiment path")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["distilbert", "gpt2", "ModernBERT", "zero-shot"],
        help="Models to analyze",
    )
    args = parser.parse_args()

    results = []
    for model in args.models:
        model_name = MODEL_NAME_MAP.get(model, model)
        new_data = {"\\textbf{Model}": f"\\textbf{{{model_name}}}"}
        for imbalance in ["05", "01", "001"]:
            results_file_path_pattern = (
                f"{args.base_path}/{model}/*/eval_outputs/"
                f"{imbalance}/random/metrics_full.json"
            )
            results_files = glob(results_file_path_pattern)
            for results_file_path in results_files:
                print(f"Processing results file: {results_file_path}")
                with open(results_file_path) as file:
                    data = json.load(file)
                # Extract relevant data

                new_data.update(
                    {
                        # "\\textbf{Accuracy}": data["accuracy"],
                        # "\\textbf{Loss}": data["loss"],
                        f"\\textbf{{Average precision}} - {imbalance}": data[
                            "average_precision"
                        ],
                        f"\\textbf{{Minority F1-Score}} - {imbalance}": data["f1_1"],
                        f"\\textbf{{Majority F1-Score}} - {imbalance}": data["f1_0"],
                    }
                )
        df = pd.DataFrame([new_data])
        results.append(df)

    # Further processing and analysis of the combined results DataFrame
    combined_results = pd.concat(results, ignore_index=True)
    print("Combined results:")
    print(combined_results)

    # Save to LaTeX table
    latex_table = combined_results.to_latex(
        index=False,
        float_format="$%.3f$",
        column_format="l" + "c" * (len(combined_results.columns) - 1),
        escape=False,
        label="tab:baseline_performance",
        caption="Summary of model performance across different architectures.",
    )
    latex_table = latex_table.replace(
        "\\begin{table}\n", "\\begin{table}[!h]\n\\centering\n"
    )

    # Make column names bold
    header_line = latex_table.split("\n")[2]  # Get the header line
    bold_header = header_line.replace("\\\\", " \\\\")
    for col in combined_results.columns:
        bold_header = bold_header.replace(col, f"\\textbf{{{col}}}")

    # Replace the header in the LaTeX table
    latex_lines = latex_table.split("\n")
    latex_lines[2] = bold_header
    latex_table_bold = "\n".join(latex_lines)

    # Save to file
    table_dir = f"{args.base_path}/tables"
    os.makedirs(table_dir, exist_ok=True)
    save_path = f"{table_dir}/model_performance_summary_new.tex"
    with open(save_path, "w") as f:
        f.write(latex_table_bold)

    print(f"LaTeX table saved to {save_path}")


if __name__ == "__main__":
    main()
