import argparse
import os
from glob import glob

import numpy as np
import pandas as pd

from arc_tigers.eval.aggregate_utils import better_than_random
from arc_tigers.eval.utils import get_metric_stats


def aggregate_strategy_results(
    base_path: str,
    models: list[str],
    imbalances: list[str],
    strategies: list[str],
    metrics: list[str] | None,
) -> None:
    """
    Iterate over combinations of models, imbalances, strategies, and metrics,
    reporting whether each strategy is better than random baseline.
    """
    rows: list[dict] = []
    # If no metrics provided, infer from random baseline stats
    if not metrics:
        # pick first model/imbalance
        sample_model = models[0]
        sample_imb = imbalances[0]
        # locate random baseline
        rand_pattern = f"{base_path}/{sample_model}/*/eval_outputs/{sample_imb}/random"
        rand_matches = glob(rand_pattern)
        if not rand_matches:
            err_msg = f"Metrics not found for {sample_model}/{sample_imb}"
            raise ValueError(err_msg)
        rand_dir = rand_matches[0]
        rand_stats, _ = get_metric_stats(rand_dir, plot=False)
        # all keys except n_labels
        metrics = [m for m in rand_stats if m != "n_labels"]
    # collect results
    for model in models:
        for imb in imbalances:
            for strat in strategies:
                for metric in metrics:
                    result_dict = better_than_random(
                        base_path, model, imb, strat, metric
                    )
                    for n, imp in result_dict.items():
                        rows.append(
                            {
                                "model": model,
                                "imbalance": imb,
                                "strategy": strat,
                                "metric": metric,
                                "n": n,
                                "percent_improvement": imp,
                            }
                        )

    n_vals = [10, 60, 110, 260, 510, 760, 1000]

    for imbalance in imbalances:
        for metric in metrics:
            df = pd.DataFrame(columns=["strategy", *n_vals])
            for strategy in strategies:
                improvement_vals = []
                for n in n_vals:
                    average_improvement = []
                    for row in rows:
                        if (
                            row["n"] == n
                            and row["strategy"] == strategy
                            and row["imbalance"] == imbalance
                            and row["metric"] == metric
                        ):
                            print(
                                f"{row['model']} {row['imbalance']} {row['metric']}: "
                                f"{row['percent_improvement']:.2f}%"
                            )
                            average_improvement.append(row["percent_improvement"])

                    print(
                        f"Average improvement for {n} {strategy}: "
                        f"{np.mean(average_improvement):.2f}%"
                    )
                    improvement_vals.append(np.mean(average_improvement))
                df.loc[len(df)] = [strategy, *improvement_vals]

            save_dir = f"tmp/tables/aggregate_performance/{imbalance}/"
            os.makedirs(save_dir, exist_ok=True)
            latex_table = df.to_latex(index=False, float_format="%.2f")
            with open(
                f"{save_dir}/{metric}.tex",
                "w",
            ) as f:
                f.write(latex_table)
            print(f"LaTeX table saved to {save_dir}/{metric}.tex")
            print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze if sampling strategies outperform random baseline"
    )
    parser.add_argument(
        "base_path",
        help="Base experiment path",
    )
    parser.add_argument(
        "--models",
        "-m",
        nargs="+",
        required=True,
        help="Model names to include",
    )
    parser.add_argument(
        "--imbalances",
        "-i",
        nargs="+",
        required=True,
        help="Test imbalance levels (e.g. '05', '01')",
    )
    parser.add_argument(
        "--strategies",
        "-s",
        nargs="+",
        required=True,
        help="Sampling strategies to evaluate",
    )
    parser.add_argument(
        "--metrics",
        "-mt",
        nargs="*",
        default=None,
        help="Metrics to compare (default: all available metrics from random baseline)",
    )
    args = parser.parse_args()

    aggregate_strategy_results(
        args.base_path,
        args.models,
        args.imbalances,
        args.strategies,
        args.metrics,
    )
