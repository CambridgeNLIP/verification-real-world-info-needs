from typing import Dict

from transformers import AutoModelForSequenceClassification

from ambifc.modeling.conf.model_config import ModelConfig


class AmbiFCModelInit:
    """
    Wrapper to initialize the correct model from within the trainer class.
    """
    def __init__(
            self,
            model_name_or_path: str,
            label2id: Dict[str, int],
            output_type: str,
            set_to_eval: bool = False
    ) -> None:
        self.set_to_eval: bool = set_to_eval
        self.model_name_or_path: str = model_name_or_path
        self.output_type: str = output_type
        self.label2id: Dict[str, int] = label2id
        self.id2label: Dict[str, int] = {label2id[k]: k for k in label2id.keys()}
        self.num_labels: int = len(self.id2label.keys())

    def model_init(self):
        if self.output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        elif self.output_type == ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id,
                problem_type="multi_label_classification"
            )
        elif self.output_type == ModelConfig.OUTPUT_DISTRIBUTION:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=self.num_labels,
                id2label=self.id2label,
                label2id=self.label2id
            )
        elif self.output_type == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name_or_path,
                num_labels=1,
                problem_type="regression"
            )
        else:
            raise NotImplementedError()

        if self.set_to_eval:
            model.eval()
        return model
