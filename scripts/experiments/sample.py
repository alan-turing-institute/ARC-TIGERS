import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from arc_tigers.data.config import HFDataConfig, load_data_config
from arc_tigers.data.utils import sample_dataset_metrics
from arc_tigers.eval.reddit_eval import get_preds
from arc_tigers.eval.utils import BiasCorrector, evaluate, get_stats
from arc_tigers.sample.acquisition import (
    AcquisitionFunctionWithEmbeds,
    SurrogateSampler,
)
from arc_tigers.sample.methods import SAMPLING_STRATEGIES
from arc_tigers.sample.utils import get_eval_outputs_dir, get_surrogate_data
from arc_tigers.training.config import TrainConfig
from arc_tigers.utils import to_json


def main(
    data_config,
    train_config,
    n_repeats,
    acq_strat,
    init_seed,
    evaluate_steps,
    output_dir,
):
    eval_data = data_config.get_test_split()

    max_labels = evaluate_steps[-1]
    if evaluate_steps[-1] > len(eval_data):
        msg = (
            f"max_labels ({max_labels}) cannot be greater than the dataset size "
            f"({len(eval_data)})."
        )
        raise ValueError(msg)

    predictions = get_preds(train_config, data_config, eval_data)

    # full dataset stats
    os.makedirs(output_dir, exist_ok=True)
    metrics = evaluate(eval_data, predictions)
    to_json(metrics, f"{output_dir}/metrics_full.json")
    stats = get_stats(predictions, eval_data["label"])
    to_json(stats, f"{output_dir}/stats_full.json")

    # Get sampler parameters
    if acq_strat not in SAMPLING_STRATEGIES:
        err_msg = (
            f"Acquisition strategy {acq_strat} not found. "
            "Available strategies: " + ", ".join(SAMPLING_STRATEGIES.keys())
        )
        raise KeyError(err_msg)
    sampler_class = SAMPLING_STRATEGIES[acq_strat]

    sampler_args = {
        "eval_data": eval_data,
        "minority_class": 1,  # Assuming class 1 is the minority class
        "model_preds": predictions,
    }

    if isinstance(data_config, HFDataConfig):
        if issubclass(sampler_class, SurrogateSampler):
            sampler_args["surrogate_train_data"] = get_surrogate_data(
                data_config, "train"
            )
        if issubclass(sampler_class, AcquisitionFunctionWithEmbeds):
            sampler_args["eval_embeds"] = get_surrogate_data(
                data_config, "test", dataset=eval_data
            ).embed

    # Start sampling
    rng = np.random.default_rng(init_seed)

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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if the output directory already exists",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for evaluation outputs, populated with a default if not set",
    )
    parser.add_argument()

    args = parser.parse_args()

    data_config = load_data_config(args.data_config)
    train_config = TrainConfig.from_path(args.train_config)

    if args.output_dir is None:
        output_dir = get_eval_outputs_dir(train_config, data_config, args.strategy)
    else:
        output_dir = args.output_dir

    if os.path.exists(output_dir):
        msg = (
            f"Output directory {output_dir} already exists. Either remove it, set "
            "--force, or change the output directory."
        )
        raise FileExistsError(msg)

    evaluate_steps = np.arange(
        args.min_labels, args.max_labels + args.eval_every, args.eval_every
    ).tolist()
    if evaluate_steps[-1] > args.max_labels:
        evaluate_steps[-1] = args.max_labels

    main(
        data_config,
        train_config,
        n_repeats=args.n_repeats,
        acq_strat=args.strategy,
        init_seed=args.seed,
        evaluate_steps=evaluate_steps,
        output_dir=output_dir,
    )
