# Adapted from https://www.philschmid.de/knowledge-distillation-bert-transformers
import torch
from transformers import Trainer


def soft_cross_entropy(predicted, target):
    assert len(predicted.shape) == 2
    assert predicted.shape == target.shape
    log_probs = torch.nn.functional.log_softmax(predicted, dim=1)
    return -(target * log_probs).sum() / predicted.shape[0]


class DistillationTrainer(Trainer):
    """
    Custom trainer for annotation distillation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        labels: torch.FloatTensor = inputs.labels
        outputs = model(**inputs)
        loss = soft_cross_entropy(outputs.logits, labels)
        if return_outputs:
            return loss, outputs
        return loss
