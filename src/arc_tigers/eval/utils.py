import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)


def evaluate(dataset, preds) -> dict[str, Any]:
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
        # Allow binary bredictions
        else:
            predictions = (logits > 0.5).astype(int)
    else:
        predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1],  # TODO: assumed labels are 0 and 1 here
    )
    acc = accuracy_score(labels, predictions)
    eval_scores = {
        "accuracy": acc,
        "f1": f1.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
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
