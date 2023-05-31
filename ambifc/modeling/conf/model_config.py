from os.path import join
from typing import Dict, Optional


class ModelConfig:
    """
    Hold all information of the model, including which task (evidence/veracity) and which outputs.
    """

    TYPE_EVIDENCE: str = 'evidence'
    TYPE_VERACITY: str = 'veracity'

    OUTPUT_SINGLE_LABEL_CLASSIFICATION: str = 'single-label-classification'
    OUTPUT_MULTI_LABEL_CLASSIFICATION: str = 'multi-label-classification'
    OUTPUT_DISTRIBUTION: str = 'distribution'
    OUTPUT_BINARY_EVIDENCE_PROBABILITY: str = 'binary-evidence-probability'

    EVIDENCE_VARIANT_BINARY: str = 'binary'
    EVIDENCE_VARIANT_STANCE: str = 'stance'

    HUMAN_DISTRIBUTION_PROBABILITY = "probability"
    HUMAN_DISTRIBUTION_SOFTMAX = "softmax"

    def __init__(self, config: Dict):
        self.config: Dict = config

        self.model_type: str = config['type']
        self.model_name: str = config['model_name']
        self.output_type: str = config['output_type']
        self.model_dest: str = config['dest']
        self.evidence_label_variant: Optional[str] = config.get('evidence_labels', None)

        self.validate()

    def get_model_task_type(self) -> str:
        """
        Get the task of the model (veracity / evidence)
        """
        return self.model_type

    def validate(self):
        assert self.model_type in {ModelConfig.TYPE_VERACITY, ModelConfig.TYPE_EVIDENCE}
        assert self.output_type in {
            ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION,
            ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION,
            ModelConfig.OUTPUT_DISTRIBUTION,
            ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY
        }, f'FOUND: {self.output_type}'
        assert self.model_name is not None
        assert self.model_dest is not None
        assert self.get_evidence_label_variant() in {
            ModelConfig.EVIDENCE_VARIANT_BINARY,
            ModelConfig.EVIDENCE_VARIANT_STANCE,
            None
        }, f'FOUND: {self.get_evidence_label_variant()}'

    def get_evidence_label_variant(self) -> Optional[str]:
        """
        Get the evidence label variant (binary /stance), or "none" for veracity models.
        """
        return self.evidence_label_variant

    def get_model_name(self) -> str:
        return self.model_name

    def get_model_dest(self) -> str:
        return self.model_dest

    def get_output_type(self) -> str:
        return self.output_type

    def get_model_dir(self) -> str:
        return join(self.config['directory'], self.config['dest'])

    def get_distribution_params(self) -> Dict:
        if not self.output_type == ModelConfig.OUTPUT_DISTRIBUTION:
            raise ValueError('get_distribution_params() output type is not set to "distribution"')

        human_distribution_method: str = self.config["distribution_params"]["human_method"]
        temperature: Optional[float] = self.config["distribution_params"]["softmax_temperature"]
        assert human_distribution_method in {
            ModelConfig.HUMAN_DISTRIBUTION_PROBABILITY, ModelConfig.HUMAN_DISTRIBUTION_SOFTMAX
        }
        return {
            "human_distribution_method": human_distribution_method,
            "temperature": temperature,
            "normalize": self.config["distribution_params"]["normalize"]
        }

    def get_confidence_evidence_params(self) -> Dict:
        assert self.output_type in {
            ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY, ModelConfig.OUTPUT_DISTRIBUTION
        }

        assert self.model_type == ModelConfig.TYPE_EVIDENCE
        return self.config['evidence_params']
