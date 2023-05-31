"""
Predict the veracity based on evidence that was selected dependant ont the veracity annotation.

Usage:
    predict_veracity_based_on_selected_evidence.py oracle <config> [--subset=<ambifc-subset>]
    """
import os
from collections import defaultdict
from copy import deepcopy
from os.path import join
from typing import Optional, List, Dict, Tuple, Iterable

import numpy as np
import torch
from datasets import Dataset
from docopt import docopt
from torch.nn import Sigmoid
from tqdm import tqdm
from transformers import Trainer, AutoTokenizer

from ambifc.modeling.conf.config import Config
from ambifc.modeling.conf.labels import get_stance_label2int, make_int2label, get_label2int
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.modeling.dataset.dataset_generators.tokenizer_map import TokenizeMap
from ambifc.modeling.dataset.dataset_generators.veracity_dataset_generator import make_veracity_label
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.modeling.prediction.sentence_predictions import is_sentence_evidence
from ambifc.modeling.training.metrics_init import AmbiFCTrainMetrics, get_ambi_metrics_for_config
from ambifc.modeling.training.model_init import AmbiFCModelInit
from ambifc.modeling.prediction.store_predictions import to_predicted_single_labels, to_probabilities_and_confidence, \
    make_predictions_from_model_probability_distribution, make_predictions_multi_label_classification, \
    to_predicted_multi_labels
from ambifc.util.fileutil import read_json, write_jsonl, write_json
import pathlib


DEFAULT_VERACITY_DEPENDENT_PREDICTION_DIR: str = join(pathlib.Path(__file__).parent.resolve(), './veracity_dependent')
DEFAULT_BASELINE_EVALUATION_DIRECTORY: str = join(
    pathlib.Path(__file__).parent.resolve(), './veracity_dependent-evaluation'
)
DEFAULT_DATA_DIR: str = './data'


def make_sample_with_sentence_keys_for_passage_dependent_sample(
        sample: Dict,
        keep_sentence_keys: List[str],
        label2int: Dict[str, int],
        output_type: str,
        include_entity_name: bool = True,
        include_section_title: bool = True,
        sep_token_for_headers: str = '@',
        distribution_params: Optional[Dict] = None
) -> Dict:
    # Add section headers in case specified.
    evidence: str = ' '.join([sample['sentences'][k] for k in keep_sentence_keys])
    if include_entity_name:
        evidence += f' {sep_token_for_headers} {sample["entity"]}'
    if include_section_title:
        evidence += f' {sep_token_for_headers} {sample["section"]}'

    result: Dict = {
        'claim_id': sample['claim_id'],
        'passage': sample['wiki_passage'],
        'used_passage_label': sample['used_passage_label'],
        'sentence_keys': list(keep_sentence_keys),

        'entity_name': sample['entity'],
        'section_title': sample['section'],
        'claim': sample['claim'],
        'evidence': evidence,
    }

    label = make_veracity_label(sample, output_type, label2int, distribution_params)
    result['label'] = label

    # Verify
    if output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
        # Need to set in manually, because the label assigning method would use the aggregated label
        # and does not consider the annotations.
        result['label'] = label2int[result['used_passage_label']]
    elif output_type == ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION:
        assert sum(result['label']) == 1, f'Expected sum of 1 but found: {result["label"]}'
        assert len(result['label']) == 3, f'Expected len of 3 but found: {result["label"]}'
    elif output_type == ModelConfig.OUTPUT_DISTRIBUTION:
        if distribution_params["human_distribution_method"] == ModelConfig.HUMAN_DISTRIBUTION_PROBABILITY:
            assert len(result['label']) == 3, f'Expected len of 3 but found: {result["label"]}'
            assert max(result['label']) == 1.0, f'Expected MAX of 1 but found: {result["label"]}'
            assert min(result['label']) == 0, f'Expected MIN of 0 but found: {result["label"]}'
        else:
            # Rest does not need to be checked -- same as above.
            assert len(result['label']) == 3, f'Expected len of 3 but found: {result["label"]}'

    return result


def create_passage_annotation_dependant_samples(sample) -> Iterable[Dict]:
    label_to_annotators = defaultdict(list)
    for annotation in sample['passage_annotations']:
        label = annotation['label']
        annotator = annotation['worker']
        label_to_annotators[label].append(annotator)

    for label in label_to_annotators.keys():

        label_sample = deepcopy(sample)
        label_sample['used_passage_label'] = label
        label_sample['passage_annotations'] = [
            ann for ann in label_sample['passage_annotations'] if ann['label'] == label
        ]
        keep_sentence_annotations = {}
        for key in label_sample['sentence_annotations'].keys():
            keep_sentence_annotations[key] = [
                ann for ann in label_sample['sentence_annotations'][key]
                if ann['annotator'] in label_to_annotators[label]
            ]
        label_sample['sentence_annotations'] = keep_sentence_annotations
        yield label_sample


def separate_data_by_veracity_prediction(samples: List[Dict]) -> List[Dict]:
    veracity_annotation_dependent_samples: List[Dict] = []
    for sample in tqdm(samples):
        veracity_annotation_dependent_samples.extend(list(create_passage_annotation_dependant_samples(sample)))
    return veracity_annotation_dependent_samples


def get_oracle_sentence_prediction_dict_for_passage_dependent_evidence(
        samples: List[Dict]
) -> Dict[Tuple[int, str, str], List[str]]:
    # Assume only AmbiFC
    return {
        (sample['claim_id'], sample['wiki_passage'], sample['used_passage_label']): sorted(list(
            filter(lambda x: is_sentence_evidence(sample, x), sample['sentence_annotations'].keys())
        ), key=lambda x: int(x)) for sample in samples
    }


def main(args) -> None:
    prediction_dest_directory: str = DEFAULT_VERACITY_DEPENDENT_PREDICTION_DIR
    ambifc_subset: str = args['--subset'] or TrainDataConfig.SUBSET_ALL_AMBIFC
    split: str = 'test'
    data_directory: str = DEFAULT_DATA_DIR

    samples: List[Dict] = get_samples_for_ambifc_subset(ambifc_subset, split, data_directory)
    samples = separate_data_by_veracity_prediction(samples)

    config: Config = Config(read_json(args['<config>']))

    if args['oracle']:
        sentence_prediction_dict: Optional[Dict[Tuple[int, str, str], List[str]]] = get_oracle_sentence_prediction_dict_for_passage_dependent_evidence(
            samples
        )
        save_name: str = f'pde.{split}.{ambifc_subset}.{config.model_config.get_model_task_type()}_oracle-ev.jsonl'
    else:
        raise NotImplementedError()

    label2int: Dict[str, int] = get_stance_label2int()
    distribution_params: Optional[Dict] = None
    if config.model_config.get_output_type() == ModelConfig.OUTPUT_DISTRIBUTION:
        distribution_params = config.model_config.get_distribution_params()

    dataset: Dataset = Dataset.from_list([
        make_sample_with_sentence_keys_for_passage_dependent_sample(
            sample,
            keep_sentence_keys=sentence_prediction_dict[sample['claim_id'], sample['wiki_passage'], sample['used_passage_label']],
            label2int=label2int,
            output_type=config.model_config.get_output_type(),
            distribution_params=distribution_params
        ) for sample in samples
    ])

    assert config.model_config.get_model_task_type() == ModelConfig.TYPE_VERACITY
    assert config.model_config.get_output_type() in {
        ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION,
        ModelConfig.OUTPUT_DISTRIBUTION,
        ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION,
    }

    ambifc_train_metrics: AmbiFCTrainMetrics = get_ambi_metrics_for_config(
        config.model_config, get_label2int(config.model_config)
    )
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(config.model_config.get_model_name())
    mapper: TokenizeMap = TokenizeMap(tokenizer)
    dataset = dataset.map(mapper.map)

    model_init = AmbiFCModelInit(
        # Load stored model
        model_name_or_path=config.model_config.get_model_dir(),
        label2id=get_label2int(config.model_config),
        output_type=config.model_config.get_output_type(),
        set_to_eval=True
    )

    trainer: Trainer = Trainer(
        model_init=model_init.model_init,
        tokenizer=tokenizer,
        # args=get_training_args(config, overwrite=False),
        compute_metrics=ambifc_train_metrics.compute_metrics
    )

    int2label: Dict[int, str] = make_int2label(label2int)
    predicted_logits, labels, metrics = trainer.predict(dataset, metric_key_prefix='pde-test-')

    if config.model_config.get_output_type() in {
        ModelConfig.OUTPUT_DISTRIBUTION, ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION
    }:
        if config.model_config.get_output_type() == ModelConfig.OUTPUT_DISTRIBUTION:
            gold_label_is_distribution: bool = True
        else:
            gold_label_is_distribution: bool = False

        predicted_logits = torch.FloatTensor(predicted_logits)
        predicted_labels: List[str] = to_predicted_single_labels(predicted_logits, int2label)
        predicted_probabilities, predicted_confidences = to_probabilities_and_confidence(predicted_logits)

        predicted_logits = predicted_logits.tolist()
        assert len(predicted_logits) == len(predicted_probabilities)
        assert len(predicted_logits) == len(predicted_confidences)

        print('predicted_probabilities', predicted_probabilities, predicted_probabilities)

        output: List[Dict] = make_predictions_from_model_probability_distribution(
            predicted_logits=predicted_logits,
            dataset=dataset,
            predicted_labels=predicted_labels,
            predicted_probabilities=predicted_probabilities,
            predicted_confidences=predicted_confidences,
            int2label=int2label,
            is_veracity_prediction=True,
            gold_label_is_distribution=gold_label_is_distribution
        )
    elif config.model_config.get_output_type() == ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION:

        sigmoid: Sigmoid = Sigmoid()
        predicted_probabilities: np.ndarray = sigmoid(torch.FloatTensor(predicted_logits)).numpy()
        predicted_labels: List[List[str]] = to_predicted_multi_labels(predicted_probabilities, int2label, threshold=0.5)
        predicted_distributions, predicted_confidences = to_probabilities_and_confidence(
            torch.FloatTensor(predicted_logits))

        predicted_logits = predicted_logits.tolist()

        assert len(predicted_logits) == len(predicted_probabilities)
        assert len(predicted_logits) == len(predicted_distributions)
        assert len(predicted_logits) == len(predicted_labels)

        output: List[Dict] = make_predictions_multi_label_classification(
            predicted_logits=predicted_logits,
            dataset=dataset,
            predicted_labels=predicted_labels,
            predicted_probabilities=predicted_probabilities,
            predicted_distributions=predicted_distributions,
            predicted_confidences=predicted_confidences,
            int2label=int2label,
            labels=labels,
            is_veracity_prediction=True
        )
    else:
        raise NotImplementedError(config.model_config.get_output_type())

    dest_pred_dir: str = join(prediction_dest_directory, config.model_config.get_model_dest())
    if not os.path.exists(dest_pred_dir):
        os.makedirs(dest_pred_dir)

    dest_name_predictions: str = join(dest_pred_dir, save_name)
    dest_name_metrics: str = join(dest_pred_dir, 'metrics.' + save_name)

    write_jsonl(dest_name_predictions, output)
    write_json(dest_name_metrics, metrics)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
