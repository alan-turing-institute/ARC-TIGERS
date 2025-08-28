import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
from scipy.stats._resampling import BootstrapResult

from arc_tigers.eval.plotting import (
    METRIC_GROUP_NAME_MAP,
    METRIC_GROUPS,
    METRIC_NAME_MAP,
    SAMPLE_STRAT_NAME_MAP,
)
from arc_tigers.eval.utils import get_se_values
from arc_tigers.utils import to_json

train_dir = "outputs/reddit_dataset_12/one-vs-all/football/42_05"
tables_dir = f"{train_dir}/tables/aggregate_replay/"
figures_dir = f"{train_dir}/figures/aggregate_replay/"
data_dir = f"{tables_dir}/raw_data/"

replays_dir = f"{train_dir}/replays"
samplers = ["accuracy_lightgbm", "info_gain_lightgbm", "minority", "ssepy"]
models = [
    ("distilbert", "default"),
    ("gpt2", "default"),
    ("ModernBERT", "short"),
    ("zero-shot", "default"),
]
imbalances = ["05", "01", "001"]

# minority_metrics = ["f1_1", "precision_1", "recall_1"]
# majority_metrics = ["f1_0", "precision_0", "recall_0"]
# overall_metrics = [
#     "accuracy",
#     "average_precision",
#     *minority_metrics,
#     *majority_metrics,
# ]


metric_keys = list(METRIC_GROUPS.keys())


def append_metric_results(
    se_history: dict[str, list], new_se: dict[str, np.ndarray]
) -> None:
    for metric in METRIC_GROUPS[metric_keys[0]]:
        if metric_keys[0] not in se_history:
            se_history[metric_keys[0]] = []
        se_history[metric_keys[0]].extend(new_se[metric].flatten().tolist())
    for metric in METRIC_GROUPS[metric_keys[1]]:
        if metric_keys[1] not in se_history:
            se_history[metric_keys[1]] = []
        se_history[metric_keys[1]].extend(new_se[metric].flatten().tolist())
    for metric in METRIC_GROUPS[metric_keys[2]]:
        if metric_keys[2] not in se_history:
            se_history[metric_keys[2]] = []
        se_history[metric_keys[2]].extend(new_se[metric].flatten().tolist())


def nan_root_mean(values):
    return np.sqrt(np.nanmean(values))


def bootstrap_rmse(
    se_values: dict[str, dict[str, list]],
) -> dict[str, dict[str, BootstrapResult]]:
    bootstrapped_results = {}
    for sampler, metrics in se_values.items():
        bootstrapped_results[sampler] = {}
        for metric_group, values in metrics.items():
            mean = nan_root_mean(values)
            bootstrapped_rmse = bootstrap(
                (values,), nan_root_mean, n_resamples=9999, method="percentile"
            )
            bootstrapped_results[sampler][metric_group] = (
                mean,
                bootstrapped_rmse.confidence_interval.low,
                bootstrapped_rmse.confidence_interval.high,
            )
            print(sampler, metric_group, bootstrapped_results[sampler][metric_group])
    return bootstrapped_results


def main():
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    self_se = {imbalance: {} for imbalance in imbalances}
    other_se = {imbalance: {} for imbalance in imbalances}
    for imbalance in imbalances:
        print("Aggregating imbalance:", imbalance)
        self_se[imbalance]["random"] = {}
        for self_model, self_config in models:
            se = get_se_values(
                str(
                    Path(train_dir)
                    / self_model
                    / self_config
                    / "eval_outputs"
                    / imbalance
                    / "random"
                )
            )
            append_metric_results(self_se[imbalance]["random"], se)

        for sampler in samplers:
            self_se[imbalance][sampler] = {}
            other_se[imbalance][sampler] = {}
            for self_model, self_config in models:
                se = get_se_values(
                    str(
                        Path(train_dir)
                        / self_model
                        / self_config
                        / "eval_outputs"
                        / imbalance
                        / sampler
                    )
                )
                append_metric_results(self_se[imbalance][sampler], se)

                for other_model, other_config in models:
                    if self_model == other_model and self_config == other_config:
                        continue
                    se = get_se_values(
                        str(
                            Path(replays_dir)
                            / imbalance
                            / f"{self_model}_{self_config}"
                            / f"from_{other_model}_{other_config}"
                            / sampler
                            / "all_seeds"
                        )
                    )
                    append_metric_results(other_se[imbalance][sampler], se)
        print("SELF")
        self_rmse = bootstrap_rmse(self_se[imbalance])
        to_json(self_rmse, f"{data_dir}self_rmse_{imbalance}.json")
        print("\nOTHER")
        other_rmse = bootstrap_rmse(other_se[imbalance])
        to_json(other_rmse, f"{data_dir}/other_rmse_{imbalance}.json")


def make_table(imbalance="05"):
    metric_keys = list(METRIC_GROUPS.keys())
    with open(f"{data_dir}/self_rmse_{imbalance}.json") as f:
        self_rmse = json.load(f)
    with open(f"{data_dir}/other_rmse_{imbalance}.json") as f:
        other_rmse = json.load(f)

    sampler = "random"
    rows = [
        {
            "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)}",
            METRIC_GROUP_NAME_MAP[
                metric_keys[0]
            ]: f"${self_rmse[sampler][metric_keys[0]][0]:.3f}^"
            f"{{{self_rmse[sampler][metric_keys[0]][2]:.3f}}}_"
            f"{{{self_rmse[sampler][metric_keys[0]][1]:.3f}}}$",
            METRIC_GROUP_NAME_MAP[
                metric_keys[1]
            ]: f"${self_rmse[sampler][metric_keys[1]][0]:.3f}^"
            f"{{{self_rmse[sampler][metric_keys[1]][2]:.3f}}}_"
            f"{{{self_rmse[sampler][metric_keys[1]][1]:.3f}}}$",
            METRIC_GROUP_NAME_MAP[
                metric_keys[2]
            ]: f"${self_rmse[sampler][metric_keys[2]][0]:.3f}^"
            f"{{{self_rmse[sampler][metric_keys[2]][2]:.3f}}}_"
            f"{{{self_rmse[sampler][metric_keys[2]][1]:.3f}}}$",
        }
    ]
    for sampler in other_rmse:
        rows.append(
            {
                "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)} (self)",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[0]
                ]: f"${self_rmse[sampler][metric_keys[0]][0]:.3f}^"
                f"{{{self_rmse[sampler][metric_keys[0]][2]:.3f}}}_"
                f"{{{self_rmse[sampler][metric_keys[0]][1]:.3f}}}$",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[1]
                ]: f"${self_rmse[sampler][metric_keys[1]][0]:.3f}^"
                f"{{{self_rmse[sampler][metric_keys[1]][2]:.3f}}}_"
                f"{{{self_rmse[sampler][metric_keys[1]][1]:.3f}}}$",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[2]
                ]: f"${self_rmse[sampler][metric_keys[2]][0]:.3f}^"
                f"{{{self_rmse[sampler][metric_keys[2]][2]:.3f}}}_"
                f"{{{self_rmse[sampler][metric_keys[2]][1]:.3f}}}$",
            }
        )
        rows.append(
            {
                "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)} (others)",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[0]
                ]: f"${other_rmse[sampler][metric_keys[0]][0]:.3f}^"
                f"{{{other_rmse[sampler][metric_keys[0]][2]:.3f}}}_"
                f"{{{other_rmse[sampler][metric_keys[0]][1]:.3f}}}$",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[1]
                ]: f"${other_rmse[sampler][metric_keys[1]][0]:.3f}^"
                f"{{{other_rmse[sampler][metric_keys[1]][2]:.3f}}}_"
                f"{{{other_rmse[sampler][metric_keys[1]][1]:.3f}}}$",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[2]
                ]: f"${other_rmse[sampler][metric_keys[2]][0]:.3f}^"
                f"{{{other_rmse[sampler][metric_keys[2]][2]:.3f}}}_"
                f"{{{other_rmse[sampler][metric_keys[2]][1]:.3f}}}$",
            }
        )

    df = pd.DataFrame(rows)
    print(df)
    df.to_latex(f"{tables_dir}/table_{imbalance}.tex", float_format="%.3f", index=False)
    df.to_csv(f"{tables_dir}/table_{imbalance}.csv", index=False)

    # Create a numeric DataFrame for plotting
    plot_rows = []
    plot_rows.append(
        {
            "Strategy": "Random",
            METRIC_GROUP_NAME_MAP[metric_keys[0]]: self_rmse["random"][metric_keys[0]][
                0
            ],
            METRIC_GROUP_NAME_MAP[metric_keys[1]]: self_rmse["random"][metric_keys[1]][
                0
            ],
            METRIC_GROUP_NAME_MAP[metric_keys[2]]: self_rmse["random"][metric_keys[2]][
                0
            ],
        }
    )
    for sampler in other_rmse:
        plot_rows.append(
            {
                "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)} (self)",
                METRIC_GROUP_NAME_MAP[metric_keys[0]]: self_rmse[sampler][
                    metric_keys[0]
                ][0],
                METRIC_GROUP_NAME_MAP[metric_keys[1]]: self_rmse[sampler][
                    metric_keys[1]
                ][0],
                METRIC_GROUP_NAME_MAP[metric_keys[2]]: self_rmse[sampler][
                    metric_keys[2]
                ][0],
            }
        )
        plot_rows.append(
            {
                "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)} (others)",
                METRIC_GROUP_NAME_MAP[metric_keys[0]]: other_rmse[sampler][
                    metric_keys[0]
                ][0],
                METRIC_GROUP_NAME_MAP[metric_keys[1]]: other_rmse[sampler][
                    metric_keys[1]
                ][0],
                METRIC_GROUP_NAME_MAP[metric_keys[2]]: other_rmse[sampler][
                    metric_keys[2]
                ][0],
            }
        )

    plot_df = pd.DataFrame(plot_rows)

    plot_df.set_index("Strategy")[METRIC_NAME_MAP[metric_keys[0]]].plot.bar()
    plt.tight_layout()
    plt.ylabel("RMSE")
    plt.title(f"{METRIC_GROUP_NAME_MAP[metric_keys[0]]} (Imbalance {imbalance})")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/minority_metrics_{imbalance}.png")
    plt.clf()
    plot_df.set_index("Strategy")[METRIC_GROUP_NAME_MAP[metric_keys[1]]].plot.bar()
    plt.ylabel("RMSE")
    plt.title(f"{METRIC_GROUP_NAME_MAP[metric_keys[1]]} (Imbalance {imbalance})")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/majority_metrics_{imbalance}.png")
    plt.clf()
    plot_df.set_index("Strategy")[METRIC_GROUP_NAME_MAP[metric_keys[2]]].plot.bar()
    plt.ylabel("RMSE")
    plt.title(f"{METRIC_GROUP_NAME_MAP[metric_keys[2]]} (Imbalance {imbalance})")
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/overall_metrics_{imbalance}.png")
    plt.clf()


def make_combined_table():
    """Create a single table combining all metrics across all imbalance levels."""
    os.makedirs(tables_dir, exist_ok=True)
    metric_keys = list(METRIC_GROUPS.keys())

    all_rows = []

    for imbalance in imbalances:
        with open(f"{data_dir}/self_rmse_{imbalance}.json") as f:
            self_rmse = json.load(f)
        with open(f"{data_dir}/other_rmse_{imbalance}.json") as f:
            other_rmse = json.load(f)

        # Add random sampler row
        sampler = "random"
        all_rows.append(
            {
                "Imbalance": imbalance,
                "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)}",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[0]
                ]: f"${self_rmse[sampler][metric_keys[0]][0]:.3f}^"
                f"{{{self_rmse[sampler][metric_keys[0]][2]:.3f}}}_"
                f"{{{self_rmse[sampler][metric_keys[0]][1]:.3f}}}$",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[1]
                ]: f"${self_rmse[sampler][metric_keys[1]][0]:.3f}^"
                f"{{{self_rmse[sampler][metric_keys[1]][2]:.3f}}}_"
                f"{{{self_rmse[sampler][metric_keys[1]][1]:.3f}}}$",
                METRIC_GROUP_NAME_MAP[
                    metric_keys[2]
                ]: f"${self_rmse[sampler][metric_keys[2]][0]:.3f}^"
                f"{{{self_rmse[sampler][metric_keys[2]][2]:.3f}}}_"
                f"{{{self_rmse[sampler][metric_keys[2]][1]:.3f}}}$",
            }
        )

        # Add other samplers (self and others)
        for sampler in other_rmse:
            all_rows.append(
                {
                    "Imbalance": imbalance,
                    "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)}",
                    METRIC_GROUP_NAME_MAP[
                        metric_keys[0]
                    ]: f"${self_rmse[sampler][metric_keys[0]][0]:.3f}^"
                    f"{{{self_rmse[sampler][metric_keys[0]][2]:.3f}}}_"
                    f"{{{self_rmse[sampler][metric_keys[0]][1]:.3f}}}$",
                    METRIC_GROUP_NAME_MAP[
                        metric_keys[1]
                    ]: f"${self_rmse[sampler][metric_keys[1]][0]:.3f}^"
                    f"{{{self_rmse[sampler][metric_keys[1]][2]:.3f}}}_"
                    f"{{{self_rmse[sampler][metric_keys[1]][1]:.3f}}}$",
                    METRIC_GROUP_NAME_MAP[
                        metric_keys[2]
                    ]: f"${self_rmse[sampler][metric_keys[2]][0]:.3f}^"
                    f"{{{self_rmse[sampler][metric_keys[2]][2]:.3f}}}_"
                    f"{{{self_rmse[sampler][metric_keys[2]][1]:.3f}}}$",
                }
            )
            all_rows.append(
                {
                    "Imbalance": imbalance,
                    "Strategy": f"{SAMPLE_STRAT_NAME_MAP.get(sampler, sampler)} "
                    "(others)",
                    METRIC_GROUP_NAME_MAP[
                        metric_keys[0]
                    ]: f"${other_rmse[sampler][metric_keys[0]][0]:.3f}^"
                    f"{{{other_rmse[sampler][metric_keys[0]][2]:.3f}}}_"
                    f"{{{other_rmse[sampler][metric_keys[0]][1]:.3f}}}$",
                    METRIC_GROUP_NAME_MAP[
                        metric_keys[1]
                    ]: f"${other_rmse[sampler][metric_keys[1]][0]:.3f}^"
                    f"{{{other_rmse[sampler][metric_keys[1]][2]:.3f}}}_"
                    f"{{{other_rmse[sampler][metric_keys[1]][1]:.3f}}}$",
                    METRIC_GROUP_NAME_MAP[
                        metric_keys[2]
                    ]: f"${other_rmse[sampler][metric_keys[2]][0]:.3f}^"
                    f"{{{other_rmse[sampler][metric_keys[2]][2]:.3f}}}_"
                    f"{{{other_rmse[sampler][metric_keys[2]][1]:.3f}}}$",
                }
            )

    df_long = pd.DataFrame(all_rows)

    # Create wide format table with imbalance levels as columns
    df_combined = df_long.pivot_table(
        index="Strategy",
        columns="Imbalance",
        values=list(METRIC_GROUP_NAME_MAP.values()),
        aggfunc="first",
    )

    # Reorder columns to group metrics by imbalance
    new_columns = []
    for imbalance in imbalances:
        new_columns.extend(
            [(metric_name, imbalance) for metric_name in METRIC_GROUP_NAME_MAP.values()]
        )

    df_combined = df_combined.reindex(columns=new_columns)

    # Flatten column names for better readability
    df_combined.columns = [
        f"{metric} - {imbalance}" for metric, imbalance in df_combined.columns
    ]

    print("Combined Table:")
    print(df_combined)

    # Save combined table
    df_combined.to_latex(
        f"{tables_dir}/combined_aggregate_replay.tex",
        float_format="%.3f",
        label="tab:combined_replay_table",
        caption=(
            "Aggregated resampling results for all samplers"
            " across different imbalances."
        ),
        column_format="l" + "c" * (len(df_combined.columns)),
    )
    df_combined.to_csv(f"{tables_dir}/combined_table.csv")

    # Also save the long format for reference
    df_long.to_latex(
        f"{tables_dir}/combined_table_long.tex", float_format="%.3f", index=False
    )
    df_long.to_csv(f"{tables_dir}/combined_table_long.csv", index=False)


if __name__ == "__main__":
    # main()
    # Create tables for each imbalance level
    for imbalance in imbalances:
        make_table(imbalance)
    # Create combined table with all metrics
    make_combined_table()
