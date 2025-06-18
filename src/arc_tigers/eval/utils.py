import logging
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from torch.nn.functional import cross_entropy

logger = logging.getLogger(__name__)


class BiasCorrector:
    """
    Implements a bias correction weighting factor for active learning sampling.

    The BiasCorrector computes a weighting factor (v_m) for a selected sample
    during active learning, as described in the formula:

        v_m = 1 + (N - M) / (N - m) * (1 / ((N - m + 1) * q_im) - 1)

    where:
        N    : Total number of samples in the dataset.
        M    : Total number of samples to be labelled (target number).
        m    : Number of samples labelled so far.
        q_im : The probability (from the acquisition function's pmf) of selecting the
        sample at step m.

    This correction is used to adjust for the sampling bias introduced by non-uniform
    acquisition functions, ensuring unbiased estimation of metrics or statistics
    over the dataset.

    Args:
        N (int): Total number of samples in the dataset.
        M (int): Total number of samples to be labelled.

    Methods:
        call(q_im, m): Computes the weighting factor for the selected sample.
    """

    def __init__(self, N, M):
        self.N = N
        self.M = M
        self.v_values = []

    def compute_weighting_factor(self, q_im, m):
        """
        Compute the weighting factor v_m for the selected sample.

        Args:
            q_im: The pmf value for the selected sample (float)
            N: Total number of samples (int)
            M: Total number of samples to be labelled (int)
            m: Number of samples labelled so far (int)

        Returns:
            v_m: Weighting factor (float)
        """
        inner = (1 / ((self.N - m + 1) * q_im)) - 1
        v_m = 1 + ((self.N - self.M) / (self.N - m)) * inner
        self.v_values.append(v_m)
        return v_m

    def apply_weighting_to_dict(
        self, q: float, m: int, metrics: dict[str, float]
    ) -> dict[str, float]:
        """
        Apply the weighting factor to all values in the metrics dictionary.

        Args:
            q: The pmf value for the selected sample (float)
            m: Number of samples labelled so far (int)
            metrics: Dictionary of metric values to be weighted {metric_name: value}

        Returns:
            dict: Dictionary with weighted metric values.
        """

        weighting = self.compute_weighting_factor(q, m)
        return {k: v * weighting for k, v in metrics.items()}


def compute_loss(
    logits: np.ndarray,
    labels: np.ndarray,
    bias_corrector: BiasCorrector | None = None,
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
    if bias_corrector:
        # apply bias correction to logits
        loss = cross_entropy(
            torch.tensor(logits, dtype=torch.float32),
            torch.tensor(labels),
            reduction="none",
        )
        loss = (loss * torch.tensor(bias_corrector.v_values, dtype=loss.dtype)).mean()
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
) -> dict[str, Any]:
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
    metrics = compute_metrics(eval_pred)
    loss = compute_loss(
        logits=preds, labels=dataset["label"], bias_corrector=bias_corrector
    )
    metrics["loss"] = loss
    metric_names = list(metrics.keys())
    for key in metric_names:
        if isinstance(metrics[key], list):
            # unpack multi-valued metrics into separate values for each class
            multi_metric = metrics.pop(key)
            if len(multi_metric) == 1:
                msg = "If metric value is a list, should have more than 1 value"
                raise ValueError(msg)
            for i, m in enumerate(multi_metric):
                metrics[f"{key}_{i}"] = m

    return metrics


# Define metrics
def compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    logits, labels = eval_pred
    if logits.ndim == 1:
        # Allow logits to be passed as the predictions
        if np.issubdtype(logits.dtype, np.integer):
            predictions = logits
        # Allow binary predictions
        else:
            predictions = (logits > 0.5).astype(int)
    else:
        predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],  # assumed binary labels (0 and 1) here
    )
    acc = accuracy_score(labels, predictions)
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


def get_stats(preds, labels):
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


def tfidf_evaluation(eval_labels, eval_predictions):
    eval_results = classification_report(
        eval_labels, eval_predictions, output_dict=True
    )
    output_dict = {}
    output_dict["eval_accuracy"] = eval_results["accuracy"]
    n_classes = len(eval_results) - 3
    output_dict["eval_f1"] = [
        eval_results[str(i)]["f1-score"] for i in range(n_classes)
    ]
    output_dict["eval_precision"] = [
        eval_results[str(i)]["precision"] for i in range(n_classes)
    ]
    output_dict["eval_recall"] = [
        eval_results[str(i)]["recall"] for i in range(n_classes)
    ]
    return output_dict
