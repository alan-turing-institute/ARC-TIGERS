import logging

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if logits.ndim > 1:
        predictions = np.argmax(logits, axis=-1)
    else:
        predictions = (logits > 0.5).astype(int)
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
