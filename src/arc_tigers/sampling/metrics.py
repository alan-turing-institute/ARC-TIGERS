import logging

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
)
from torch.nn.functional import cross_entropy

from arc_tigers.sampling.bias import BiasCorrector

logger = logging.getLogger(__name__)


def compute_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | list[float] | None = None,
) -> float:
    """
    Compute the cross-entropy loss between logits and labels.

    Args:
        logits: logits from the model, shape (n_samples, n_classes)
        labels: labels for the dataset, shape (n_samples,)

    Returns:
        loss: The computed cross-entropy loss.
    """
    # compute cross-entropy loss
    if sample_weight:
        # apply bias correction to logits
        loss = cross_entropy(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(labels),
            reduction="none",
        )
        loss = (loss * torch.tensor(sample_weight, dtype=loss.dtype)).mean()
    else:
        loss = cross_entropy(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(labels),
            reduction="mean",
        )
    return loss.item()


def evaluate(
    dataset,
    preds,
    bias_corrector: BiasCorrector | None = None,
) -> dict[str, int | float]:
    """
    Compute metrics for a given dataset with preds.

    Args:
        dataset: The dataset to compute metrics for
        preds: The predictions for the dataset.
        model: The model to compute metrics with.

    Returns:
        A dictionary containing the computed metrics.
    """
    # Placeholder for actual metric computation
    eval_pred = (preds, dataset["label"])
    metrics = compute_metrics(eval_pred, bias_corrector=bias_corrector)
    single_value_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, list):
            # unpack multi-valued metrics into separate values for each class
            if len(value) == 1:
                msg = "If metric value is a list, should have more than 1 value"
                raise ValueError(msg)
            for i, m in enumerate(value):
                single_value_metrics[f"{key}_{i}"] = m
        else:
            single_value_metrics[key] = value

    return single_value_metrics


# Define metrics
def compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
    bias_corrector: BiasCorrector | None = None,
) -> dict[str, int | float | list[int | float]]:
    logits, labels = eval_pred
    if logits.ndim == 1:
        # Allow logits to be passed as the predictions
        if np.issubdtype(logits.dtype, np.integer):
            predictions = logits
            probs = logits  # fallback, not ideal
        else:
            predictions = (logits > 0.5).astype(int)
            probs = logits
    else:
        probs = softmax(logits)[:, 1] if logits.shape[1] > 1 else softmax(logits)[:, 0]
        predictions = np.argmax(logits, axis=-1)

    sample_weight = bias_corrector.v_values if bias_corrector else None
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],  # assumed binary labels (0 and 1) here
        sample_weight=sample_weight,
    )
    acc = accuracy_score(
        labels,
        predictions,
        sample_weight=sample_weight,
    )

    loss = compute_loss(logits=logits, labels=labels, sample_weight=sample_weight)

    # Improved Average Precision (AP) handling
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        avg_precision = float("nan")
        warning = (
            "average_precision_score is undefined: only one class"
            f" ({unique_labels[0]}) present in labels."
        )
        logger.warning(warning)
    else:
        avg_precision = average_precision_score(
            labels, probs, sample_weight=sample_weight
        )

    # no. of samples per class in the eval data (also assumed binary labels here,
    # minlength should be set to the number of classes to ensure
    # len(samples_per_class) == n_classes)
    if isinstance(labels, torch.Tensor):
        n_class = torch.bincount(labels, minlength=2)
    else:
        n_class = np.bincount(labels, minlength=2)

    eval_scores = {
        "accuracy": acc,
        "f1": f1.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "loss": loss,
        "average_precision": avg_precision,
        "n_class": n_class.tolist(),
    }
    logger.info("Eval metrics: %s", eval_scores)
    return eval_scores


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute the softmax of the logits.

    Args:
        logits: The logits to compute the softmax for.

    Returns:
        The softmax of the logits.
    """
    return np.exp(logits) / np.sum(np.exp(logits), axis=-1, keepdims=True)


def entropy(logits: np.ndarray) -> np.ndarray:
    """
    Compute the entropy of the logits.

    Args:
        logits: The logits to compute the entropy for.

    Returns:
        The entropy of the logits.
    """
    softmax_probs = softmax(logits)
    return -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)


def per_sample_stats(preds, labels):
    """
    Compute statistics for a given dataset with preds.

    Args:
        preds: The labels for the dataset.
        labels: The dataset to compute metrics for

    Returns:
        A dictionary containing the computed metrics.
    """
    # Placeholder for actual metric computation
    logits = np.array(preds)
    class_preds = np.argmax(logits, axis=-1)
    label_vector = np.array(labels)

    accuracy_vector = np.array(class_preds == label_vector, dtype=int)
    entropy_values = entropy(logits)

    return {
        "accuracy": accuracy_vector.tolist(),
        "softmax": softmax(logits).tolist(),
        "entropy": entropy_values.tolist(),
    }
