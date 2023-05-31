from typing import List, Dict, Iterable, Optional, Union

from datasets import Dataset
from transformers import AutoTokenizer

from ambifc.modeling.conf.labels import get_evidence_label_to_int, make_int2label
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.dataset.dataset_generators.tokenizer_map import TokenizeMap
from ambifc.util.label_util import sentence_annotations_to_binary, sentence_annotations_to_stance, \
    sentence_annotation_to_binary_evidence_confidence, make_soft_label_probability_distribution, \
    make_soft_label_softmax_distribution


class EvidenceDatasetGenerator:
    """
    Generator to create a dataset based on individual evidence sentences.
    """
    def __init__(
            self,
            output_type: str,
            samples: List[Dict],
            tokenizer: AutoTokenizer,
            include_entity_name: bool,
            include_section_title: bool,
            evidence_label_variant: str,
            sep_token_for_headers: str = '@',
            params: Optional[Dict] = None
    ):
        self.output_type: str = output_type
        self.samples: List[Dict] = samples
        self.tokenizer: AutoTokenizer = tokenizer
        self.include_entity_name: bool = include_entity_name
        self.include_section_title: bool = include_section_title
        self.label2int: Dict[str, int] = get_evidence_label_to_int(evidence_label_variant)
        self.int2label: Dict[int, str] = make_int2label(self.label2int)
        self.evidence_label_variant: str = evidence_label_variant
        self.sep_token_for_headers: str = sep_token_for_headers
        self.params: Optional[Dict] = params

        assert self.output_type in {
            ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION,
            ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY,
            ModelConfig.OUTPUT_DISTRIBUTION
        }

        if self.output_type == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
            assert self.evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_BINARY
        if self.output_type == ModelConfig.OUTPUT_DISTRIBUTION:
            assert self.evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_STANCE
            assert self.params is not None and 'human_distribution_method' in self.params

    def get_dataset(self) -> Dataset:
        """
        Get a dataset based on the samples provided to the generator.
        """
        sentence_samples: List[Dict] = list(self._extract_sentences(self.samples))
        dataset: Dataset = Dataset.from_list(sentence_samples)
        mapper: TokenizeMap = TokenizeMap(self.tokenizer, evidence_key='sentence')
        return dataset.map(mapper.map)

    def _extract_sentences(self, samples: Iterable[Dict]) -> Iterable[Dict]:
        for sample in samples:

            # Only consider actual sentences, i.e. excluding empty strings.
            # Consider all sentences in their order.
            sentence_keys: List[str] = sorted(list(sample['labels']['sentences'].keys()), key=lambda x: int(x))
            sentence_keys = list(filter(lambda key: len(sample['sentences'][key].strip()) > 0, sentence_keys))

            for sentence_key in sentence_keys:

                # Label may be different based on the model (distribution, single-label / probability)
                label: Union[str, float, List[float]] = self._get_label(sample['sentence_annotations'][sentence_key])

                sentence: str = sample['sentences'][sentence_key]

                # If specified, attach information of the Wikipedia entity and section.
                if self.include_entity_name:
                    sentence += f' {self.sep_token_for_headers} {sample["entity"]}'
                if self.include_section_title:
                    sentence += f' {self.sep_token_for_headers} {sample["section"]}'

                # Construct sentence-level instance
                return_sample: Dict = {
                    'claim_id': sample['claim_id'],
                    'passage': sample['wiki_passage'],
                    'sentence_key': sentence_key,

                    'entity_name': sample['entity'],
                    'section_title': sample['section'],
                    'claim': sample['claim'],
                    'sentence': sentence
                }

                # Assign label and do some validation
                if self.output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
                    assert type(label) == str
                    return_sample['label'] = self.label2int[label]
                elif self.output_type == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
                    assert type(label) == float
                    return_sample['label'] = label
                elif self.output_type == ModelConfig.OUTPUT_DISTRIBUTION:
                    assert type(label) == list
                    return_sample['label'] = label
                else:
                    raise NotImplementedError(self.output_type)

                yield return_sample

    def _get_label(self, annotations: List[Dict]) -> Union[str, float, List[float]]:
        """
        Assign a label based on the annotations.
        """

        # Get all evidence annotations of a given sentence.
        annotations: List[str] = list(map(lambda x: x['annotation'], annotations))

        # For single-label: Select binary or stance
        if self.output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
            if self.evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_BINARY:
                return sentence_annotations_to_binary(annotations)
            elif self.evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_STANCE:
                return sentence_annotations_to_stance(annotations)
            else:
                raise NotImplementedError(self.evidence_label_variant)

        # For probability labels
        elif self.output_type == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
            assert self.evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_BINARY
            return sentence_annotation_to_binary_evidence_confidence(annotations)

        # For annotation distribution labels
        elif self.output_type == ModelConfig.OUTPUT_DISTRIBUTION:
            if self.params["human_distribution_method"] == ModelConfig.HUMAN_DISTRIBUTION_PROBABILITY:
                label: List[float] = make_soft_label_probability_distribution(annotations, self.int2label)
            elif self.params["human_distribution_method"] == ModelConfig.HUMAN_DISTRIBUTION_SOFTMAX:
                temperature: float = self.params['temperature']
                normalize: bool = self.params['normalize']
                label: List[float] = make_soft_label_softmax_distribution(
                    annotations, self.int2label, temperature, normalize
                )
            else:
                raise NotImplementedError(self.params["human_distribution_method"])
            return label
        else:
            raise NotImplementedError(self.output_type)
