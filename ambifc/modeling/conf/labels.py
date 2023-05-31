from typing import Dict, List

from ambifc.modeling.conf.model_config import ModelConfig


def get_label2int(model_config: ModelConfig) -> Dict[str, int]:
    """
    Get a dictionary that maps labels to integers for training based on the configuration.

    :param model_config: Defines model, outputs and task.
    """
    if model_config.get_model_task_type() == ModelConfig.TYPE_VERACITY:
        return get_stance_label2int()
    else:
        assert model_config.get_model_task_type() == ModelConfig.TYPE_EVIDENCE
        evidence_label_variant: str = model_config.get_evidence_label_variant()
        return get_evidence_label_to_int(evidence_label_variant)


def get_evidence_label_to_int(evidence_label_variant: str) -> Dict[str, int]:
    """
    Geta dictionary that maps evidence labels to integers for binary/stance labels.
    """
    if evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_STANCE:
        return get_stance_label2int()
    elif evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_BINARY:
        return get_binary_evidence_label2int()
    else:
        raise NotImplementedError()


def get_evidence_conversion_dict() -> Dict[str, str]:
    """
    Get a dictionary that converts evidence stance labels to binary labels.
    """
    return {
        'neutral': 'neutral',
        'supporting': 'evidence',
        'refuting': 'evidence'
    }


def get_full_neutral_distribution(int2lbl: Dict[int, str]) -> List[float]:
    """
    Get a distribution that is 100% neutral. This can be used, for example, when no evidence was selected and the
    prediction defaults to neutral.
    """
    return [
        0. if int2lbl[i] != 'neutral' else 1.0
        for i in sorted(list(int2lbl.keys()))
    ]


def get_binary_evidence_label2int() -> Dict[str, int]:
    """
    Get the dictionary that maps binary evidence labels to integers.
    """
    return {
        'neutral': 0,
        'evidence': 1
    }


def get_stance_label2int() -> Dict[str, int]:
    """
    Get the dictionary that maps stance evidence/veracity labels to integers.
    """
    return {
        'refuting': 0,
        'neutral': 1,
        'supporting': 2
    }


def make_label_list(label2int: Dict[str, int]) -> List[str]:
    """
    Get a list of labels sorted byx their integer value.
    """
    int2label: Dict[int, str] = make_int2label(label2int)
    return [int2label[key] for key in sorted(list(int2label.keys()))]


def make_int2label(label2int: Dict[str, int]) -> Dict[int, str]:
    """
    Convert a dictionary to map any integer value to the respective string label.
    """
    return {
        label2int[k]: k for k in label2int
    }
