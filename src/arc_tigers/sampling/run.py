import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from arc_tigers.data.config import HFDataConfig
from arc_tigers.samplers.acquisition import (
    AcquisitionFunctionWithEmbeds,
    SurrogateSampler,
)
from arc_tigers.samplers.methods import SAMPLING_STRATEGIES
from arc_tigers.sampling.bias import BiasCorrector
from arc_tigers.sampling.metrics import evaluate, per_sample_stats
from arc_tigers.sampling.utils import (
    get_preds,
    get_surrogate_data,
    sample_dataset_metrics,
)
from arc_tigers.utils import to_json


def run_sampling(
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
    full_metrics = evaluate(eval_data, predictions)
    to_json(full_metrics, f"{output_dir}/metrics_full.json")
    stats = per_sample_stats(predictions, eval_data["label"])
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
