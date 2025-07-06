import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm

from arc_tigers.data.config import HFDataConfig, load_data_config
from arc_tigers.data.utils import sample_dataset_metrics
from arc_tigers.eval.reddit_eval import get_preds
from arc_tigers.eval.utils import BiasCorrector, evaluate, get_stats
from arc_tigers.sample.methods import SAMPLING_STRATEGIES
from arc_tigers.training.config import TrainConfig
from arc_tigers.utils import create_dirs, to_json


def main(
    n_repeats,
    eval_data,
    surrogate_data,
    acq_strat,
    predictions,
    init_seed,
    max_labels,
    evaluate_steps,
    embed_dir,
):
    rng = np.random.default_rng(init_seed)

    if acq_strat not in SAMPLING_STRATEGIES:
        err_msg = (
            f"Acquisition strategy {acq_strat} not found. "
            "Available strategies: " + ", ".join(SAMPLING_STRATEGIES.keys())
        )
        raise KeyError(err_msg)
    sampler_class = SAMPLING_STRATEGIES[acq_strat]
    sampler_args = {
        "eval_data": eval_data,
        "surrogate_data": surrogate_data,
        "embed_dir": embed_dir,
        "minority_class": 1,  # Assuming class 1 is the minority class
        "model_preds": predictions,
    }

    # full dataset stats
    metrics = evaluate(eval_data, predictions)
    to_json(metrics, f"{output_dir}/metrics_full.json")
    stats = get_stats(preds, eval_data["label"])
    to_json(stats, f"{output_dir}/stats_full.json")

    # iteratively sample dataset and compute metrics, repeated n_repeats times
    max_labels = int(max_labels) if max_labels < len(predictions) else len(predictions)

    for _ in tqdm(range(n_repeats)):
        seed = rng.integers(1, 2**32 - 1)  # Generate a random seed
        sampler_args["seed"] = seed
        acq_func = sampler_class(**sampler_args)
        bias_corrector = (
            BiasCorrector(N=len(predictions), M=max_labels)
            if acq_strat != "random"
            else None
        )
        metrics = sample_dataset_metrics(
            eval_data,
            predictions,
            acq_func,
            max_labels=max_labels,
            evaluate_steps=evaluate_steps,
            bias_corrector=bias_corrector,
        )

        pd.DataFrame(metrics).to_csv(f"{output_dir}/metrics_{seed}.csv", index=False)
        to_json(
            {
                "sample_idx": acq_func.labelled_idx,
                "sample_prob": acq_func.sample_prob,
                "dataset_size": len(predictions),
                "n_labels": max_labels,
            },
            f"{output_dir}/sample_{seed}.json",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier")
    parser.add_argument(
        "data_config",
        help="path to the data config file specifying the test set to use",
    )
    parser.add_argument(
        "train_config",
        help="path to the training config file specifying the trained model to use",
    )
    parser.add_argument("strategy", type=str, help="Sampling strategy")

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

    output_dir, predictions_dir, _ = create_dirs(
        save_dir=args.save_dir,
        data_config_path=args.data_config,
        acq_strat=args.acq_strat,
        class_balance=args.class_balance,
    )

    data_config = load_data_config(args.data_config)
    train_config = TrainConfig.from_path(args.train_config)

    eval_data = data_config.get_test_split()
    preds = get_preds(train_config, data_config, eval_data)

    surrogate_train_data = (
        data_config.get_train_split()
        if isinstance(data_config, HFDataConfig) and args.acq_strat != "random"
        else None
    )

    main(
        n_repeats=args.n_repeats,
        eval_data=eval_data,
        surrogate_data=surrogate_train_data,
        acq_strat=args.acq_strat,
        predictions=preds,
        init_seed=args.seed,
        max_labels=args.max_labels,
        evaluate_steps=np.arange(
            args.min_labels, args.max_labels, args.eval_every
        ).tolist(),
        embed_dir=predictions_dir,
    )
