from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


class EvaluatorClaimSentenceBinary:
    """
    Used during training.
    """
    def __init__(self, encoder, batch_size):
        self.batch_size = batch_size
        self.encoder = encoder

    def evaluate(self, model, dataset):
        model.eval()
        targets = []
        outputs = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=self.batch_size):
                encoded_dict = self.encoder.encode(batch['claim'], batch['sentence'])
                logits = model(**encoded_dict)[0]
                targets.extend(batch['label'].float().tolist())
                outputs.extend(logits.argmax(dim=1).tolist())

        metrics = {
            'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
            'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
            'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
            'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
        }

        return metrics, metrics['macro_f1']


class EvaluatorClaimSentence3Way:

    def __init__(self, encoder, batch_size, key_evidence='sentence'):
        self.batch_size = batch_size
        self.encoder = encoder
        self.key_evidence = key_evidence

    def evaluate(self, model, dataset):
        model.eval()
        targets = []
        outputs = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=self.batch_size):
                encoded_dict = self.encoder.encode(batch['claim'], batch[self.key_evidence])
                logits = model(**encoded_dict)[0]
                targets.extend(batch['label'].float().tolist())
                outputs.extend(logits.argmax(dim=1).tolist())

        result = {
            'macro_f1': f1_score(targets, outputs, zero_division=0, average='macro'),
            'f1': tuple(f1_score(targets, outputs, zero_division=0, average=None)),
            'precision': tuple(precision_score(targets, outputs, zero_division=0, average=None)),
            'recall': tuple(recall_score(targets, outputs, zero_division=0, average=None))
        }
        return result, result['macro_f1']