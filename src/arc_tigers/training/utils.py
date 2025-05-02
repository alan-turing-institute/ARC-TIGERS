import torch
from torch import nn
from transformers import Trainer

from arc_tigers.utils import get_device


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        assert "loss_weights" in kwargs, "loss_weights must be provided"
        self.loss_weights = kwargs.pop("loss_weights", None)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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
