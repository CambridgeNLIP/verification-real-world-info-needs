from typing import Dict, List, Optional, Iterable, Tuple, Union

from datasets import Dataset
from transformers import AutoTokenizer

from ambifc.modeling.conf.labels import get_stance_label2int, make_label_list, make_int2label
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.conf.train_config import TrainConfig
from ambifc.modeling.dataset.dataset_generators.tokenizer_map import TokenizeMap
import random

from ambifc.util.label_util import make_soft_label_probability_distribution, make_soft_label_softmax_distribution


def get_evidence_sentence_keys(sample: Dict, sentence_keys: List[str], inverse: bool = False) -> List[str]:
    """
    Get all sentence keys of evidence sentences.
    :param sample: The instance containing all sentences.
    :param sentence_keys: A list of all sentence keys.
    :param inverse: If set to true, all non-evidence sentences will be selected.
    """

    def is_evidence_sentence(key: str) -> bool:
        # Remove all neutral annotations
        annotations: Iterable[str] = map(lambda x: x['annotation'], sample['sentence_annotations'][key])
        annotations = filter(lambda x: x != 'neutral', annotations)

        # If annotations remain, consider the sentence as evidence.
        if inverse:
            return len(list(annotations)) == 0
        else:
            return len(list(annotations)) > 0

    return list(filter(is_evidence_sentence, sentence_keys))


def make_veracity_label(
        sample: Dict, output_type: str, label2int: Dict[str, int], distribution_params: Optional[Dict]
) -> Union[List[float], int]:
    """
    Create the veracity label for a given sample based on the output type.
    :param sample: Veracity label will be created for this instance.
    :param output_type: The desired output type (single-label / distribution / ...).
    :param label2int: Maps each label to an integer value.
    :param distribution_params: Optional additional parameters. Needed when the output is a distribution.
    """

    # Single-Label Classification: return the integer of the assigned label via Dawid Skene.
    if output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
        return label2int[sample['labels']['passage']]

    # Multi-Label-Classification: Assign each label that is included in the veracity annotations.
    elif output_type == ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION:
        label_list: List[str] = make_label_list(label2int)
        passage_annotations: List[str] = list(map(lambda x: x['label'], sample['passage_annotations']))
        return list(map(lambda x: 1.0 if x in passage_annotations else 0.0, label_list))

    # Annotator Distribution: Return the annotator distribution (computed as defined in the parameters)
    else:
        assert output_type == ModelConfig.OUTPUT_DISTRIBUTION
        assert distribution_params is not None
        passage_annotations: List[str] = list(map(lambda x: x['label'], sample['passage_annotations']))
        int2label: Dict[int, str] = make_int2label(label2int)
        if distribution_params["human_distribution_method"] == ModelConfig.HUMAN_DISTRIBUTION_PROBABILITY:
            return make_soft_label_probability_distribution(passage_annotations, int2lbl=int2label)
        else:
            assert distribution_params["human_distribution_method"] == ModelConfig.HUMAN_DISTRIBUTION_SOFTMAX
            temperature: Optional[float] = distribution_params['temperature']
            normalize: bool = distribution_params['normalize']
            return make_soft_label_softmax_distribution(passage_annotations, int2label, temperature, normalize)


class VeracityDatasetGenerator:
    """
    Generator for a dataset for veracity prediction.
    """
    def __init__(
            self,
            samples: List[Dict],
            tokenizer: AutoTokenizer,
            include_entity_name: bool,
            include_section_title: bool,
            evidence_sampling_strategy: str,
            classification_output_type: str,
            include_empty_sentences: bool = False,
            sample_sentences_min: Optional[int] = None,
            sample_sentences_max: Optional[int] = None,
            sep_token_for_headers: str = '@',
            sample_seed: Optional[int] = None,
            distribution_params: Optional[Dict] = None
    ):
        self.samples: List[Dict] = samples
        self.tokenizer: AutoTokenizer = tokenizer
        self.include_entity_name: bool = include_entity_name
        self.include_section_title: bool = include_section_title
        self.evidence_sampling_strategy: str = evidence_sampling_strategy
        self.include_empty_sentences: bool = include_empty_sentences
        self.label2int: Dict[str, int] = get_stance_label2int()
        self.int2label: Dict[int, str] = make_int2label(self.label2int)
        self.label_list: List[str] = make_label_list(self.label2int)
        self.sep_token_for_headers: str = sep_token_for_headers
        self.classification_output_type: str = classification_output_type
        self.distribution_params: Optional[Dict] = distribution_params

        assert self.classification_output_type in {
            ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION,
            ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION,
            ModelConfig.OUTPUT_DISTRIBUTION
        }

        assert self.evidence_sampling_strategy in {
            TrainConfig.EVIDENCE_SELECTION_FULL,
            TrainConfig.EVIDENCE_SELECTION_ORACLE,
            TrainConfig.EVIDENCE_SELECTION_SAMPLE_IF_NO_EVIDENCE_FOR_NEUTRAL,
            TrainConfig.SAMPLE_ALWAYS_NEUTRAL_EVIDENCE
        }

        if self.evidence_sampling_strategy in {
            TrainConfig.EVIDENCE_SELECTION_SAMPLE_IF_NO_EVIDENCE_FOR_NEUTRAL,
            TrainConfig.SAMPLE_ALWAYS_NEUTRAL_EVIDENCE
        }:
            assert sample_sentences_min is not None and sample_sentences_min >= 0
            assert sample_sentences_max is not None and sample_sentences_max >= 0
            assert sample_seed is not None
            self.sample_sentences_min: int = sample_sentences_min
            self.sample_sentences_max: int = sample_sentences_max
            random.seed(sample_seed)

    def get_datasets_without_given_evidence(self) -> Dataset:
        """
        Use this method if no preselected evidence should be included. Evidence will then be samples based on the
        evidence selection strategy.
        """
        preprocessed_samples: List[Dict] = list(self._extract_samples(
            sentence_predictions=None, sentence_prediction_source=None
        ))

        dataset: Dataset = Dataset.from_list(preprocessed_samples)

        mapper: TokenizeMap = TokenizeMap(self.tokenizer)
        return dataset.map(mapper.map)

    def get_dataset_based_on_sentence_predictions(
            self,
            predictions: Dict[Tuple[int, str], List[str]],
            sentence_prediction_source: str
    ) -> Dataset:
        """
        Only use the selected sentences as evidence.
        :param predictions: Map each claim identified by claim_id and passage_id to a list of selected sentences.
        :param sentence_prediction_source: Name of the sentence prediction source. Will be included in the prediction
                                           file for additional information.
        """
        assert predictions is not None
        assert sentence_prediction_source is not None

        preprocessed_samples: List[Dict] = list(self._extract_samples(
            sentence_predictions=predictions, sentence_prediction_source=sentence_prediction_source
        ))
        dataset: Dataset = Dataset.from_list(preprocessed_samples)

        mapper: TokenizeMap = TokenizeMap(self.tokenizer)
        return dataset.map(mapper.map)

    def _extract_samples(
            self,
            sentence_predictions: Optional[Dict[Tuple[int, str], List[str]]],
            sentence_prediction_source: Optional[str]
    ) -> Iterable[Dict]:
        for sample in self.samples:

            # Collect all possible sentences for the sample (excluding empty strings unless specified otherwise).
            sentence_keys: Iterable[str] = sorted(list(sample['labels']['sentences'].keys()), key=lambda x: int(x))
            if not self.include_empty_sentences:
                sentence_keys = filter(lambda key: len(sample['sentences'][key]) > 0, sentence_keys)

            # If sentence predictions are provided use them.
            # Usually used during inference when predicting a pipeline system based on
            # the predictions of an evidence selection model.
            if sentence_predictions is not None:
                predicted_evidence_keys: List[str] = sentence_predictions[(sample['claim_id'], sample['wiki_passage'])]
                yield self._make_sample_with_sentence_keys(
                    sample,
                    keep_sentence_keys=predicted_evidence_keys,
                    other_keys={
                        'sentence_prediction_from': sentence_prediction_source
                    }
                )

            # If NO evidence sentences are provided: They will be samples / all will be used
            else:
                # Use the full document
                if self.evidence_sampling_strategy == TrainConfig.EVIDENCE_SELECTION_FULL:
                    yield self._make_sample_with_sentence_keys(sample, keep_sentence_keys=list(sentence_keys))

                # Sample evidence (for training)
                else:
                    sentence_keys: List[str] = list(sentence_keys)
                    using_sentence_key_lists: List[Tuple[List[str], Optional[Dict]]] = []

                    # Sampling will be based on the actual evidence sentences.
                    evidence_sentence_keys: List[str] = get_evidence_sentence_keys(sample, sentence_keys)

                    # We always include evidence sentences in training (optimal for a perfect evidence selection)
                    if len(evidence_sentence_keys) > 0:
                        using_sentence_key_lists.append((evidence_sentence_keys, None))

                    # If no evidence sentences exist, non-evidence sentences will be samples.
                    # If evidence sentences exist, but the sampling strategy dictates that none-evidence sentences will
                    # be sampled either way, non-evidence sentences will be samples.
                    has_no_evidence: bool = len(evidence_sentence_keys) == 0
                    if has_no_evidence or self.evidence_sampling_strategy == TrainConfig.SAMPLE_ALWAYS_NEUTRAL_EVIDENCE:
                        no_evidence_sentence_keys: List[str] = get_evidence_sentence_keys(
                            sample, sentence_keys, inverse=True
                        )

                        # We can only use it if only-neutral sentences exist
                        if len(no_evidence_sentence_keys) > 0:

                            # Select a random number of non-evidence sentences to keep.
                            max_sentences: int = min([self.sample_sentences_max, len(no_evidence_sentence_keys)])
                            min_sentences: int = min([self.sample_sentences_min, len(no_evidence_sentence_keys)])
                            num_sentences: int = random.randint(min_sentences, max_sentences)
                            indices: List[int] = list(range(len(no_evidence_sentence_keys)))
                            random.shuffle(indices)
                            indices = sorted(indices[:num_sentences])

                            # Add a sample based on non-evidence sentences and force the label to neutral.
                            keep_sentence_keys: List[str] = list(map(lambda i: no_evidence_sentence_keys[i], indices))
                            using_sentence_key_lists.append((keep_sentence_keys, {
                                # forcibly set to neutral
                                'label': self._get_neutral_label(),
                                'set_label_to_neutral': True
                            }))

                    assert len(using_sentence_key_lists) > 0

                    # We now may have more than one sample for each original sample (hence a list), IF
                    # neutral samples are created with or without existing evidence.
                    for current_sentence_keys, additional_keys in using_sentence_key_lists:

                        # other_keys override previous keys from the sample.
                        yield self._make_sample_with_sentence_keys(
                            sample, current_sentence_keys, other_keys=additional_keys
                        )

    def _make_sample_with_sentence_keys(
            self, sample: Dict, keep_sentence_keys: List[str], other_keys: Optional[Dict] = None
    ) -> Dict:
        """
        Create a sample for the dataset based on preselected sentence keys.
        :param sample: The sample that will be transformed to be in the dataset.
        :param keep_sentence_keys: The list of sentence keys that will be kept as evidence.
        :param other_keys: A dictionary with additional keys that override keys in the sample.
        """

        evidence: str = ' '.join([sample['sentences'][k] for k in keep_sentence_keys])

        # Add information about Wikipedia section and entity if specified.
        if self.include_entity_name:
            evidence += f' {self.sep_token_for_headers} {sample["entity"]}'
        if self.include_section_title:
            evidence += f' {self.sep_token_for_headers} {sample["section"]}'

        result: Dict = {
            'claim_id': sample['claim_id'],
            'passage': sample['wiki_passage'],
            'sentence_keys': list(keep_sentence_keys),

            'entity_name': sample['entity'],
            'section_title': sample['section'],
            'claim': sample['claim'],
            'evidence': evidence,

            'label': self._make_passage_label(sample)
        }

        if other_keys is not None:
            for key in other_keys:
                result[key] = other_keys[key]

        return result

    def _make_passage_label(self, sample: Dict) -> Union[int, Union[List[int], List[float], List[List[float]]]]:
        return make_veracity_label(sample, self.classification_output_type, self.label2int, self.distribution_params)

    def _get_neutral_label(self):
        """
        Create a neutral label based on the desired output type.
        """
        if self.classification_output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
            return self.label2int['neutral']
        elif self.classification_output_type == ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION:
            return list(map(lambda x: 1.0 if x == 'neutral' else 0.0, self.label_list))
        elif self.classification_output_type == ModelConfig.OUTPUT_DISTRIBUTION:

            # Assume 2 annotations as this is the most frequent case for neutral samples.
            num_annotations: int = 2
            if self.distribution_params["human_distribution_method"] == ModelConfig.HUMAN_DISTRIBUTION_PROBABILITY:
                return make_soft_label_probability_distribution(['neutral'] * num_annotations, int2lbl=self.int2label)
            elif self.distribution_params["human_distribution_method"] == ModelConfig.HUMAN_DISTRIBUTION_SOFTMAX:
                temperature: Optional[float] = self.distribution_params['temperature']
                normalize: bool = self.distribution_params['normalize']
                return make_soft_label_softmax_distribution(
                    ['neutral'] * num_annotations, self.int2label, temperature, normalize
                )
            else:
                raise NotImplementedError(self.distribution_params["human_distribution_method"])
        else:
            raise NotImplementedError(self.classification_output_type)
