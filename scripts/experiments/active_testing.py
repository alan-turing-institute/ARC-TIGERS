import argparse
import json
import os
from collections.abc import Iterable
from typing import cast

import numpy as np
import pandas as pd
from tqdm import tqdm

from arc_tigers.data.utils import sample_dataset_metrics
from arc_tigers.eval.reddit_eval import get_preds, get_train_data_from_exp_dir
from arc_tigers.eval.utils import BiasCorrector, evaluate, get_stats
from arc_tigers.sample.acquisition import (
    AccSampler,
    AcquisitionFunction,
    DistanceSampler,
    InformationGainSampler,
    IsolationForestSampler,
)
from arc_tigers.utils import create_dir


def main(
    output_dir,
    n_repeats,
    data_dict,
    acq_strat,
    predictions,
    init_seed,
    max_labels,
    evaluate_steps,
):
    evaluation_dataset = data_dict["evaluation_dataset"]
    bias_correction = False

    rng = np.random.default_rng(init_seed)
    if acq_strat == "distance":
        sampler_class = DistanceSampler
    elif acq_strat == "random_forest_acc":
        sampler_class = AccSampler
        bias_correction = True
    elif acq_strat == "random_forest_ig":
        sampler_class = InformationGainSampler
    elif acq_strat == "iForest":
        sampler_class = IsolationForestSampler
        bias_correction = True
    else:
        # raise error if acq_strat is not one of the available strategies
        # uses the __str__ method of the sampler classes to get the available strategies
        err_msg = f"Unknown acquisition strategy: {acq_strat}. Available strategies: "
        # get the names of the available acquisition strategies and join them
        err_msg += ", ".join(
            [
                strat.name
                for strat in cast(
                    Iterable[AcquisitionFunction],
                    [DistanceSampler, AccSampler, InformationGainSampler],
                )
            ]
        )
        raise ValueError(err_msg)

    # full dataset stats
    metrics = evaluate(evaluation_dataset, preds)
    stats = get_stats(preds, evaluation_dataset["label"])

    with open(f"{output_dir}/metrics_full.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{output_dir}/stats_full.json", "w") as f:
        json.dump(stats, f, indent=2)
    # iteratively sample dataset and compute metrics, repeated n_repeats times

    # set max_labels to the minimum of max_labels and the number of predictions
    max_labels = int(max_labels) if max_labels < len(predictions) else len(predictions)
    for _ in tqdm(range(n_repeats)):
        seed = rng.integers(1, 2**32 - 1)  # Generate a random seed
        acq_func = sampler_class(
            data=data_dict, sampling_seed=seed, eval_dir=output_dir
        )
        if hasattr(sampler_class, "set_model_preds"):
            acq_func.set_model_preds(predictions)
        if hasattr(sampler_class, "surrogate_pretrain"):
            acq_func.surrogate_pretrain()

        metrics = sample_dataset_metrics(
            evaluation_dataset,
            predictions,
            acq_func,
            max_labels=max_labels,
            evaluate_steps=evaluate_steps,
            bias_corrector=BiasCorrector(
                N=len(preds),
                M=max_labels,
            )
            if bias_correction
            else None,
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

    output_dir = create_dir(
        save_dir=args.save_dir,
        data_config_path=args.data_config,
        acq_strat=args.acq_strat,
        class_balance=args.class_balance,
    )

    if os.path.isfile(output_dir + "predictions.npy"):
        print("loading saved predictions..")
        preds = np.load(output_dir + "predictions.npy")
        _, eval_data = get_preds(
            data_config_path=args.data_config,
            model_config_path=args.model_config,
            save_dir=args.save_dir,
            class_balance=args.class_balance,
            seed=args.seed,
            synthetic_args=None,
            preds_exist=True,
        )
    else:
        preds, eval_data = get_preds(
            data_config_path=args.data_config,
            model_config_path=args.model_config,
            save_dir=args.save_dir,
            class_balance=args.class_balance,
            seed=args.seed,
            synthetic_args=None,
        )
        print("saving predictions..")
        np.save(output_dir + "predictions.npy", preds)

    data_dict = {
        "evaluation_dataset": eval_data,
        "surrogate_train_dataset": get_train_data_from_exp_dir(exp_dir=args.save_dir),
    }

    main(
        output_dir=output_dir,
        n_repeats=args.n_repeats,
        data_dict=data_dict,
        acq_strat=args.acq_strat,
        predictions=preds,
        init_seed=args.seed,
        max_labels=args.max_labels,
        evaluate_steps=np.arange(
            args.min_labels, args.max_labels, args.eval_every
        ).tolist(),
    )
