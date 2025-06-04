import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from arc_tigers.data.utils import sample_dataset_metrics
from arc_tigers.eval.reddit_eval import get_preds
from arc_tigers.eval.utils import evaluate, get_stats
from arc_tigers.sample.acquisition import DistanceSampler


def main(
    save_dir,
    n_repeats,
    dataset,
    acq_strat,
    predictions,
    init_seed,
    max_labels,
    evaluate_steps,
    class_balance,
):
    rng = np.random.default_rng(init_seed)
    if class_balance != 1.0:
        output_dir = (
            f"{save_dir}/imbalanced_{acq_strat}_sampling_outputs_"
            f"{str(class_balance).replace('.', '')}/"
        )
    else:
        output_dir = f"{save_dir}/{acq_strat}_sampling_outputs/"

    if acq_strat == "distance":
        sampler = DistanceSampler
    os.makedirs(output_dir, exist_ok=True)
    # full dataset stats
    metrics = evaluate(dataset, preds)
    stats = get_stats(preds, dataset["label"])

    with open(f"{output_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{output_dir}/stats_full.json", "w") as f:
        json.dump(stats, f, indent=2)
    # iteratively sample dataset and compute metrics, repeated n_repeats times
    for _ in tqdm(range(n_repeats)):
        seed = rng.integers(1, 2**32 - 1)  # Generate a random seed
        acq_func = sampler(data=dataset, sampling_seed=seed)
        metrics = sample_dataset_metrics(
            dataset,
            predictions,
            acq_func,
            max_labels=max_labels,
            evaluate_steps=evaluate_steps,
        )
        pd.DataFrame(metrics).to_csv(f"{output_dir}/metrics_{seed}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument(
        "data_config",
        help=(
            "path to the data config yaml file, or 'synthetic' to generate a "
            "synthetic dataset"
        ),
    )
    parser.add_argument(
        "model_config",
        help=(
            "path to the model config yaml file, or 'beta_model' to use a "
            "synthetic model. model_adv must be set if using beta_model."
        ),
    )
    parser.add_argument(
        "save_dir",
        type=str,
        default=None,
        help="Path to save the model and results",
    )
    parser.add_argument(
        "acq_strat",
        type=str,
        default=None,
        help="Acquisition function to use",
    )

    parser.add_argument(
        "--class_balance",
        type=float,
        default=1.0,
        help="Balance between the classes",
    )
    parser.add_argument("--n_repeats", type=int, required=True)
    parser.add_argument("--max_labels", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=50,
        help="Re-compute metrics every eval_every samples",
    )
    parser.add_argument(
        "--min_labels",
        type=int,
        default=10,
        help="Minimum number of labels to sample before computing metrics",
    )

    args = parser.parse_args()

    preds, test_dataset = get_preds(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        save_dir=args.save_dir,
        class_balance=args.class_balance,
        seed=args.seed,
        synthetic_args=None,
    )
    print("obtained predictions")

    main(
        save_dir=args.save_dir,
        n_repeats=args.n_repeats,
        dataset=test_dataset,
        acq_strat=args.acq_strat,
        predictions=preds,
        init_seed=args.seed,
        max_labels=args.max_labels,
        evaluate_steps=np.arange(
            args.min_labels, args.max_labels, args.eval_every
        ).tolist(),
        class_balance=args.class_balance,
    )
