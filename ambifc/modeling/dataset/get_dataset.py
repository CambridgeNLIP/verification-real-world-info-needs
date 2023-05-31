from typing import Dict, Optional, List, Tuple

from datasets import Dataset
from transformers import AutoTokenizer

from ambifc.modeling.conf.config import Config
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.dataset.dataset_generators.evidence_dataset_generator import EvidenceDatasetGenerator
from ambifc.modeling.dataset.dataset_generators.veracity_dataset_generator import VeracityDatasetGenerator


def get_dataset(
        samples: List[Dict],
        config: Config,
        tokenizer: AutoTokenizer,
        sentence_prediction: Optional[Dict[Tuple[int, str], List[str]]] = None,
        sentence_prediction_source: Optional[str] = None
) -> Dataset:
    """
    Get a dataset for training or evaluation given the provided samples
    :param samples:
        Samples that are within the dataset.
    :param config:
        Will define what kind of dataset.
        (such as sentence or passage level, single or multi label, binary or stance evidence, ...)
    :param tokenizer:
        Used AutoTokenizer
    :param sentence_prediction:
        Optional: They are only required for veracity prediction during inference. Dictionary containing the evidence
        sentences that must be used when predicting the veracity.
    :param sentence_prediction_source:
        Optional: They are only required if sentence_prediction is given. A string to document where the evidence was
        taken from (e.g. from which evidence-selection model).
    :return:
    """
    task_type: str = config.model_config.get_model_task_type()
    if task_type == ModelConfig.TYPE_VERACITY:

        # Only needed for sampling evidence in the neutral case.
        sample_seed: int = config.training_config.get_seed()
        return get_dataset_for_veracity_prediction(
            samples, config, tokenizer,
            sentence_predictions=sentence_prediction,
            sentence_prediction_source=sentence_prediction_source,
            seed=sample_seed
        )
    else:
        return get_dataset_for_evidence_prediction(samples, config, tokenizer)


def get_dataset_for_evidence_prediction(
    samples: List[Dict],
    config: Config,
    tokenizer: AutoTokenizer,
) -> Dataset:
    """
    Get a dataset for evidence selection (T1).

    :param samples: Use the provided samples.
    :param config: Config of the experiment defines how samples must be processed.
    :param tokenizer: To tokenize claim and evidence.
    """
    output_type: str = config.model_config.get_output_type()
    evidence_label_variant: str = config.model_config.get_evidence_label_variant()

    if output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
        assert evidence_label_variant is not None
        generator: EvidenceDatasetGenerator = EvidenceDatasetGenerator(
            output_type=output_type,
            samples=samples,
            tokenizer=tokenizer,
            include_entity_name=config.train_data_config.is_include_entity_name(),
            include_section_title=config.train_data_config.is_include_section_header(),
            evidence_label_variant=config.model_config.get_evidence_label_variant()
        )
        return generator.get_dataset()
    elif output_type == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
        assert evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_BINARY
        generator: EvidenceDatasetGenerator = EvidenceDatasetGenerator(
            output_type=output_type,
            samples=samples,
            tokenizer=tokenizer,
            include_entity_name=config.train_data_config.is_include_entity_name(),
            include_section_title=config.train_data_config.is_include_section_header(),
            evidence_label_variant=config.model_config.get_evidence_label_variant()
        )
        return generator.get_dataset()
    elif output_type == ModelConfig.OUTPUT_DISTRIBUTION:
        assert evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_STANCE
        distribution_params: Dict = config.model_config.get_distribution_params()
        generator: EvidenceDatasetGenerator = EvidenceDatasetGenerator(
            output_type=output_type,
            samples=samples,
            tokenizer=tokenizer,
            include_entity_name=config.train_data_config.is_include_entity_name(),
            include_section_title=config.train_data_config.is_include_section_header(),
            evidence_label_variant=config.model_config.get_evidence_label_variant(),
            params=distribution_params
        )
        return generator.get_dataset()
    else:
        raise NotImplementedError(output_type)


def get_dataset_for_veracity_prediction(
        samples: List[Dict],
        config: Config,
        tokenizer: AutoTokenizer,
        sentence_predictions: Optional[Dict[Tuple[int, str], List[str]]] = None,
        sentence_prediction_source: Optional[str] = None,
        seed: Optional[int] = None
) -> Dataset:
    """
    Get a dataset for veracity prediction (T2).

    :param samples: Use the provided samples.
    :param config: Config of the experiment defines how samples must be processed.
    :param tokenizer: To tokenize claim and evidence.
    :param sentence_predictions: Optional sentence predictions. If set, the selected sentences will be used as evidence.
    :param sentence_prediction_source: Indicates the source of the selected sentences.
    :param seed: random seed needed when evidence sentences must be sampled.
    """

    if config.model_config.get_output_type() == ModelConfig.OUTPUT_DISTRIBUTION:
        distribution_params: Dict = config.model_config.get_distribution_params()
    else:
        distribution_params: Optional[Dict] = None

    min_sample_sentences, max_sample_sentences = config.training_config.get_min_max_sampling_sentences()
    generator: VeracityDatasetGenerator = VeracityDatasetGenerator(
        samples=samples,
        tokenizer=tokenizer,
        include_entity_name=config.train_data_config.is_include_entity_name(),
        include_section_title=config.train_data_config.is_include_section_header(),
        evidence_sampling_strategy=config.training_config.get_evidence_sampling_strategy(),
        sample_sentences_min=min_sample_sentences,
        sample_sentences_max=max_sample_sentences,
        sample_seed=seed,
        classification_output_type=config.model_config.get_output_type(),
        distribution_params=distribution_params
    )

    if sentence_predictions is None:
        # Evidence sentence must be selected based on defined strategy.
        return generator.get_datasets_without_given_evidence()
    else:
        # Evidence sentences are preselected based on the predictions.
        return generator.get_dataset_based_on_sentence_predictions(
            predictions=sentence_predictions,
            sentence_prediction_source=sentence_prediction_source
        )
