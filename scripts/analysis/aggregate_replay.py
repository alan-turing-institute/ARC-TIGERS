import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import bootstrap
from scipy.stats._resampling import BootstrapResult

from arc_tigers.eval.utils import get_se_values
from arc_tigers.utils import to_json

train_dir = "/Users/jroberts/data/outputs/reddit_dataset_12/one-vs-all/football/42_05"
replays_dir = f"{train_dir}/replays"
samplers = ["accuracy_lightgbm", "info_gain_lightgbm", "minority"]
models = [
    ("distilbert", "default"),
    ("gpt2", "default"),
    ("ModernBERT", "fp16"),
    ("zero-shot", "default"),
]
imbalances = ["05", "01", "001"]

minority_metrics = ["f1_1", "precision_1", "recall_1"]
majority_metrics = ["f1_0", "precision_0", "recall_0"]
overall_metrics = [
    "accuracy",
    "average_precision",
    *minority_metrics,
    *majority_metrics,
]


def append_metric_results(
    se_history: dict[str, list], new_se: dict[str, np.ndarray]
) -> None:
    for metric in minority_metrics:
        se_history["minority"].extend(new_se[metric].flatten().tolist())
    for metric in majority_metrics:
        se_history["majority"].extend(new_se[metric].flatten().tolist())
    for metric in overall_metrics:
        se_history["overall"].extend(new_se[metric].flatten().tolist())


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
    self_se = {}
    other_se = {}
    self_se["random"] = {
        "minority": [],
        "majority": [],
        "overall": [],
    }
    for self_model, self_config in models:
        for imbalance in imbalances:
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
            append_metric_results(self_se["random"], se)

    for sampler in samplers:
        self_se[sampler] = {
            "minority": [],
            "majority": [],
            "overall": [],
        }
        other_se[sampler] = {
            "minority": [],
            "majority": [],
            "overall": [],
        }
        for self_model, self_config in models:
            for imbalance in imbalances:
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
                append_metric_results(self_se[sampler], se)

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
                    append_metric_results(other_se[sampler], se)

    print("SELF")
    self_rmse = bootstrap_rmse(self_se)
    to_json(self_rmse, "self_rmse.json")
    print()
    print("OTHER")
    other_rmse = bootstrap_rmse(other_se)
    to_json(other_rmse, "other_rmse.json")


def make_table():
    with open("self_rmse.json") as f:
        self_rmse = json.load(f)
    with open("other_rmse.json") as f:
        other_rmse = json.load(f)

    sampler = "random"
    rows = [
        {
            "sampler": f"{sampler}",
            "Minority Metrics": self_rmse[sampler]["minority"][0],
            "Majority Metrics": self_rmse[sampler]["majority"][0],
            "All Metrics": self_rmse[sampler]["overall"][0],
        }
    ]
    for sampler in other_rmse:
        rows.append(
            {
                "sampler": f"{sampler} (self)",
                "Minority Metrics": self_rmse[sampler]["minority"][0],
                "Majority Metrics": self_rmse[sampler]["majority"][0],
                "All Metrics": self_rmse[sampler]["overall"][0],
            }
        )
        rows.append(
            {
                "sampler": f"{sampler} (others)",
                "Minority Metrics": other_rmse[sampler]["minority"][0],
                "Majority Metrics": other_rmse[sampler]["majority"][0],
                "All Metrics": other_rmse[sampler]["overall"][0],
            }
        )

    df = pd.DataFrame(rows)
    print(df)
    df.to_latex("table.tex", float_format="%.3f", index=False)
    df.to_csv("table.csv", index=False)
    df.set_index("sampler")["Minority Metrics"].plot.bar()
    plt.tight_layout()
    plt.ylabel("RMSE")
    plt.title("Minority Metrics")
    plt.tight_layout()
    plt.savefig("minority_metrics.png")
    df.set_index("sampler")["Majority Metrics"].plot.bar()
    plt.ylabel("RMSE")
    plt.title("Majority Metrics")
    plt.tight_layout()
    plt.savefig("majority_metrics.png")
    df.set_index("sampler")["All Metrics"].plot.bar()
    plt.ylabel("RMSE")
    plt.title("All Metrics")
    plt.tight_layout()
    plt.savefig("overall_metrics.png")


if __name__ == "__main__":
    # main()
    make_table()
