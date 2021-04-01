import torch
from typing import Dict, Optional
from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models import BasicClassifier, Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import F1Measure


@Model.register("simple_perclass")
class SimpleClassifierWithPerClassSores(BasicClassifier):

    def __init__(self, vocab: Vocabulary, text_field_embedder: TextFieldEmbedder, seq2vec_encoder: Seq2VecEncoder,
                 seq2seq_encoder: Seq2SeqEncoder = None, feedforward: Optional[FeedForward] = None,
                 dropout: float = None, num_labels: int = None, label_namespace: str = "labels",
                 namespace: str = "tokens", initializer: InitializerApplicator = InitializerApplicator(), **kwargs) -> None:
        super().__init__(vocab, text_field_embedder, seq2vec_encoder, seq2seq_encoder, feedforward, dropout, num_labels,
                         label_namespace, namespace, initializer, **kwargs)

        self._f1_true = F1Measure(self.vocab.get_token_index("true", label_namespace))
        self._f1_false = F1Measure(self.vocab.get_token_index("false", label_namespace))

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        a = self._f1_true.get_metric(reset)
        b = self._f1_false.get_metric(reset)

        true_p, true_r, true_f1 = a["precision"],a["recall"],a["f1"]
        false_p, false_r, false_f1 = b["precision"],b["recall"],b["f1"]

        metrics = {
            "accuracy": self._accuracy.get_metric(reset),
            "true_p": true_p,
            "true_r": true_r,
            "true_f1": true_f1,
            "false_p": false_p,
            "false_r": false_r,
            "false_f1": false_f1,
        }
        return metrics

    def forward(  # type: ignore
        self, tokens: TextFieldTensors, label: torch.IntTensor = None
    ) -> Dict[str, torch.Tensor]:
        output_dict = super().forward(tokens,label)

        if label is not None:
            self._f1_true(output_dict['logits'], label)
            self._f1_false(output_dict['logits'], label)

        return output_dict