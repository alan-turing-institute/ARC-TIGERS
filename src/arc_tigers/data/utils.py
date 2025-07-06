from copy import deepcopy

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer

from arc_tigers.eval.utils import BiasCorrector, evaluate
from arc_tigers.sample.acquisition import AcquisitionFunction

EXPECTED_KEYS = {"text", "label", "len"}


def hf_train_test_split(dataset: Dataset, **split_kwargs) -> tuple[Dataset, Dataset]:
    """Splits a dataset into train and test sets, returning them as a tuple rather than
    a dict like the default HF datasets `train_test_split` method."""
    split = dataset.train_test_split(**split_kwargs)
    return split["train"], split["test"]


def tokenize_data(data: Dataset, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Loads and preprocesses the Reddit dataset based on the specified configuration.

    Returns:
        tuple: A tuple containing the training dataset, evaluation dataset, and test
        dataset.
    """
    return data.map(
        lambda batch: tokenizer(batch["text"], padding=True, truncation=True),
        batched=True,
    )


def get_target_mapping(eval_task: str, target_subreddits: list[str]) -> dict[str, int]:
    if eval_task == "multi-class":
        return {subreddit: index for index, subreddit in enumerate(target_subreddits)}
    if eval_task == "one-vs-all" or eval_task == "data-drift":
        return dict.fromkeys(target_subreddits, 1)
    err_msg = f"Invalid evaluation task: {eval_task}"
    raise ValueError(err_msg)


def sample_dataset_metrics(
    dataset: Dataset,
    preds: np.ndarray,
    sampler: AcquisitionFunction,
    evaluate_steps: list[int],
    bias_corrector: BiasCorrector | None = None,
) -> list[dict[str, float]]:
    """
    Simulate iteratively random sampling the whole dataset, re-computing metrics
    after each sample.

    Args:
        dataset: The dataset to sample from.
        preds: The predictions for the dataset.
        model: The model to compute metrics with.
        seed: The random seed for sampling.
        max_labels: The maximum number of labels to sample. If None, the whole dataset
            will be sampled.

    Returns:
        A list of dictionaries containing the computed metrics after each sample.
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
