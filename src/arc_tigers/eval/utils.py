import logging

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logger = logging.getLogger(__name__)


# Define metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
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
