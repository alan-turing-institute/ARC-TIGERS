from collections import Counter

import torch
from torch import nn
from transformers import Trainer

from arc_tigers.utils import get_device


class WeightedLossTrainer(Trainer):
    """
    Custom trainer, inhereted from the base Trainer class, that uses a weighted cross
    entropy loss function for training. This is useful for imbalanced datasets.
    The weights are calculated by the inverse of the class frequencies.

    Args:
        loss_weights (list): A list of weights for each class. The length of the list
        must be equal to the number of classes.
        args: Additional arguments to pass to the base Trainer class.
        kwargs: Additional keyword arguments to pass to the base Trainer class.
    """

    def __init__(self, *args, **kwargs) -> None:
        assert "loss_weights" in kwargs, "loss_weights must be provided"
        self.loss_weights = kwargs.pop("loss_weights", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs) -> tuple:
        """
        computes the loss for the inputs and the model

        Args:
            model: Huggingface model to be used for training
            inputs: the inputs to the model
            return_outputs: whether to return the outputs of the model.
            Defaults to False.

        Returns:
            tuple: A tuple containing the loss and the outputs of the model if
            return_outputs is True, otherwise just the loss.
        """
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(
            weight=torch.tensor(self.loss_weights).to(get_device())
        )
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def get_label_weights(dataset, verbose=True):
    label_counter = Counter(dataset["label"])
    label_counts = dict(sorted(label_counter.items(), key=lambda item: item[1]))
    label_weights = [
        1 - (label_count / sum(label_counts.values()))
        for label_count in label_counts.values()
    ]

    if verbose:
        print("Label counts in the training dataset:")
        for label_idx, (label, count) in enumerate(label_counts.items()):
            print(f"Label: {label}, Count: {count}, Weight: {label_weights[label_idx]}")

    return label_weights, label_counts
