import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from tqdm import tqdm

from arc_tigers.data.config import HFDataConfig, SyntheticDataConfig
from arc_tigers.samplers.acquisition import (
    AcquisitionFunctionWithEmbeds,
    SurrogateSampler,
)
from arc_tigers.samplers.fixed import FixedSampler
from arc_tigers.samplers.methods import SAMPLING_STRATEGIES
from arc_tigers.samplers.sampler import Sampler
from arc_tigers.sampling.bias import BiasCorrector
from arc_tigers.sampling.metrics import evaluate, per_sample_stats
from arc_tigers.sampling.utils import get_preds, get_surrogate_data
from arc_tigers.training.config import TrainConfig
from arc_tigers.utils import to_json


def sample_dataset_metrics(
    dataset: Dataset,
    preds: np.ndarray,
    sampler: Sampler,
    evaluate_steps: list[int],
    bias_corrector: BiasCorrector | None = None,
) -> list[dict[str, float]]:
    """
    Simulate iteratively random sampling the whole dataset, re-computing metrics
    after each sample.

    Args:
        dataset: The dataset to sample from.
        preds: The predictions for the dataset.
        sampler: The sampler to use for sampling.
        evaluate_steps: Steps at which to compute metrics.
        bias_corrector: Optional bias corrector to adjust metrics based on sampling
            probabilities.

    Returns:
        A list of dictionaries containing the computed metrics after each step in
        evaluate_steps.
    """
    max_labels = evaluate_steps[-1]
    evaluate_steps = deepcopy(evaluate_steps)
    metrics = []
    next_eval_step = evaluate_steps.pop(0)
    for n in tqdm(range(max_labels)):
        _, q = sampler.sample()
        if bias_corrector is not None:
            bias_corrector.compute_weighting_factor(q_im=q, m=n + 1)
        if (n + 1) == next_eval_step:
            metric = evaluate(
                dataset[sampler.labelled_idx],
                preds[sampler.labelled_idx],
                bias_corrector=bias_corrector,
            )
            metric["n"] = n + 1
            metrics.append(metric)
            if evaluate_steps:
                next_eval_step = evaluate_steps.pop(0)
            else:
                break

    return metrics


def sampling_loop(
    data_config: HFDataConfig | SyntheticDataConfig,
    train_config: TrainConfig,
    n_repeats: int,
    sampling_strategy: str,
    init_seed: int,
    evaluate_steps: list[int],
    output_dir: str | Path,
    retrain_every: int,
    surrogate_pretrain: bool,
    replay_sample_indices: list[int] | None = None,
    replay_sample_probabilities: list[float] | None = None,
):
    """
    Repeat sampling and evaluation on a dataset a specified number of times.

    Args:
        data_config: Specifies test dataset to use.
        train_config: Specifies trained model to use.
        n_repeats: Number of times to repeat the sampling and evaluation.
        sampling_strategy: The sampling strategy to use.
        init_seed: Initial seed for random number generation.
        evaluate_steps: Steps at which to compute metrics.
        output_dir: Directory to save evaluation outputs.
        retrain_every: Surrogate update frequency, if used.
        surrogate_pretrain: If True, pre-train surrogate model on the training set.
        replay_sample_indices: Optional list of sample indices for replay experiments.
        replay_sample_probabilities: Optional list of sample probabilities for replay.
    """
    eval_data = data_config.get_test_split()

    max_labels = evaluate_steps[-1]
    if evaluate_steps[-1] > len(eval_data):
        msg = (
            f"max_labels ({max_labels}) cannot be greater than the dataset size "
            f"({len(eval_data)})."
        )
        raise ValueError(msg)

    predictions = get_preds(train_config, data_config, eval_data)

    os.makedirs(output_dir, exist_ok=True)
    config = {
        "data_config": data_config.config_name,
        "train_config": train_config.config_name,
        "sampling_strategy": sampling_strategy,
        "n_repeats": n_repeats,
        "max_labels": max_labels,
        "init_seed": init_seed,
        "evaluate_steps": evaluate_steps,
        "retrain_every": retrain_every,
    }
    with open(f"{output_dir}/config.yaml", "w") as f:
        yaml.safe_dump(config, f)

    # full dataset stats
    full_metrics = evaluate(eval_data, predictions)
    to_json(full_metrics, f"{output_dir}/metrics_full.json")
    stats = per_sample_stats(predictions, eval_data["label"])
    to_json(stats, f"{output_dir}/stats_full.json")

    # Get sampler parameters
    if sampling_strategy not in SAMPLING_STRATEGIES:
        err_msg = (
            f"Acquisition strategy {sampling_strategy} not found. "
            "Available strategies: " + ", ".join(SAMPLING_STRATEGIES.keys())
        )
        raise KeyError(err_msg)
    sampler_class = SAMPLING_STRATEGIES[sampling_strategy]

    # Check if this is a replay experiment
    is_replay = (
        replay_sample_indices is not None and replay_sample_probabilities is not None
    )

    if is_replay:
        # For replay experiments, use FixedSampler regardless of sampling_strategy
        sampler_class = FixedSampler
        sampler_args = {
            "eval_data": eval_data,
            "seed": init_seed,
            "sample_indices": replay_sample_indices,
            "sample_probabilities": replay_sample_probabilities,
        }
    else:
        # Standard sampler arguments for non-replay experiments
        sampler_args = {
            "eval_data": eval_data,
            "minority_class": 1,  # Assuming class 1 is the minority class
            "model_preds": predictions,
            "retrain_every": retrain_every,
        }

    if isinstance(data_config, HFDataConfig) and not is_replay:
        if issubclass(sampler_class, SurrogateSampler) and surrogate_pretrain:
            sampler_args["surrogate_train_data"] = get_surrogate_data(
                data_config, "train"
            )
        if issubclass(sampler_class, AcquisitionFunctionWithEmbeds):
            sampler_args["eval_embeds"] = get_surrogate_data(
                data_config, "test", dataset=eval_data
            ).embed

    # Start sampling
    rng = np.random.default_rng(init_seed)

    # For replay experiments, only run once with the predetermined sequence
    actual_n_repeats = 1 if is_replay else n_repeats

    for _ in tqdm(range(actual_n_repeats)):
        # For replay, use original seed; for normal experiments, generate random seed
        seed = init_seed if is_replay else rng.integers(1, 2**32 - 1)

        if not is_replay:
            sampler_args["seed"] = seed

        sampler = sampler_class(**sampler_args)
        bias_corrector = (
            BiasCorrector(N=len(predictions), M=max_labels)
            if sampling_strategy != "random"
            else None
        )
        metrics = sample_dataset_metrics(
            eval_data,
            predictions,
            sampler,
            evaluate_steps=evaluate_steps,
            bias_corrector=bias_corrector,
        )

        pd.DataFrame(metrics).to_csv(f"{output_dir}/metrics_{seed}.csv", index=False)
        to_json(
            {
                "sample_idx": sampler.labelled_idx,
                "sample_prob": sampler.sample_prob,
                "dataset_size": len(predictions),
                "n_labels": max_labels,
            },
            f"{output_dir}/sample_{seed}.json",
        )
