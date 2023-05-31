"""
Applies majority based veracity prediction using the sentence labels.

Usage:
    predict_baseline_distributions.py full-labels <label> [<label2>] [<label3>] [--subset=<ambifc-subset>]
    predict_baseline_distributions.py avg-dist <subset> [--subset=<ambifc-subset>]
    """

from collections import Counter
from copy import copy
from os.path import join
from typing import List, Dict

import numpy as np
from docopt import docopt

from pass_eval_ambifc import evaluate_all_veracity_prediction
from ambifc.modeling.conf.labels import get_stance_label2int, make_int2label
from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.modeling.prediction.make_multi_label_predictions import make_multi_label_predictions_from_distribution
from ambifc.util.fileutil import write_jsonl_to_dir
import pathlib


DEFAULT_BASELINE_PREDICTION_DIRECTORY: str = join(
    pathlib.Path(__file__).parent.resolve(), './veracity_baselines_distributions'
)
DEFAULT_BASELINE_EVALUATION_DIRECTORY: str = join(
    pathlib.Path(__file__).parent.resolve(), './veracity_baselines_dist-evaluation'
)
DEFAULT_DATA_DIR: str = './data'


def get_passage_annotation_distribution(sample: Dict, int2lbl: Dict[int, str]) -> List[float]:
    counts = Counter([ann['label'] for ann in sample['passage_annotations']])
    return [counts.get(int2lbl[i], 0) / len(sample['passage_annotations']) for i in range(len(int2lbl.keys()))]


def get_distribution_for_labels(labels: List[str], int2lbl: Dict[int, str]) -> List[float]:
    return [1/len(labels) if int2lbl[i] in labels else 0.0 for i in range(len(int2lbl))]


def make_prediction_average_distribution(distribution: List[float], int2lbl: Dict[int, str]):
    return {
        'sentence_keys': [],
        'logits': distribution,
        'predicted_distribution': distribution,
        'is_evidence_based_prediction': False,
        'predicted': 'neutral',
        'predicted_confidence': max(distribution),
        'multi_predicted': make_multi_label_predictions_from_distribution(int2lbl, distribution)
    }


def make_prediction_with_full_labels(labels: List[str], int2lbl: Dict[int, str]):
    dist: List[float] = get_distribution_for_labels(labels, int2lbl)
    return {
        'sentence_keys': [],
        'logits': dist,
        'predicted_distribution': dist,
        'is_evidence_based_prediction': False,
        'predicted': 'neutral',
        'predicted_confidence': max(dist),
        'multi_predicted': labels
    }


def make_distribution_baseline_predictions(
        eval_samples: List[Dict],
        baseline_variant: str,
        params: Dict = None
) -> List[Dict]:
    result: List[Dict] = []
    int2lbl: Dict[int, str] = make_int2label(get_stance_label2int())

    for sample in eval_samples:
        if baseline_variant == 'full-labels':
            pred: Dict = make_prediction_with_full_labels(params['labels'], int2lbl)
        elif baseline_variant == 'avg-dist':
            pred = make_prediction_average_distribution(params['distribution'], int2lbl)
        else:
            raise NotImplementedError(baseline_variant)
        sample = copy(sample)
        sample = {
            'claim_id': sample['claim_id'],
            'passage': sample['wiki_passage'],
            'entity_name': sample['entity'],
            'section_title': sample['section'],
            'claim': sample['claim'],
            'sentence_prediction_from': None,
        }
        for key in pred:
            sample[key] = pred[key]

        result.append(sample)
    return result


def main(args) -> None:
    prediction_dest_directory: str = DEFAULT_BASELINE_PREDICTION_DIRECTORY
    ambifc_subset: str = args['--subset'] or TrainDataConfig.SUBSET_ALL_AMBIFC
    split: str = 'test'
    data_directory: str = DEFAULT_DATA_DIR

    samples: List[Dict] = get_samples_for_ambifc_subset(ambifc_subset, split, data_directory)

    if args['full-labels']:
        # Set everything to a single label
        labels = sorted([
            args[key] for key in ['<label>', '<label2>', '<label3>'] if key in args and args[key] is not None
        ])
        lbl2int: Dict[str, int] = get_stance_label2int()
        assert set(labels) | set(lbl2int.keys()) == set(lbl2int.keys()), f'{labels}'

        predictions: List[Dict] = make_distribution_baseline_predictions(samples, 'full-labels', {
            'labels': labels
        })
        file_name: str = f'full-labels__{"-".join(labels)}.predictions.{ambifc_subset}.{split}.jsonl'
    elif args['avg-dist']:
        tuning_samples: List[Dict] = get_samples_for_ambifc_subset(args['<subset>'], 'train', data_directory)
        lbl2int: Dict[str, int] = get_stance_label2int()
        distributions = np.array([
            get_passage_annotation_distribution(sample, make_int2label(lbl2int)) for sample in tuning_samples
        ])

        predictions: List[Dict] = make_distribution_baseline_predictions(samples, 'avg-dist', {
            'distribution': list(map(float, np.mean(distributions, axis=0)))
        })
        file_name: str = f'avg-dist__from-{args["<subset>"]}.predictions.{ambifc_subset}.{split}.jsonl'
    else:
        raise NotImplementedError()

    write_jsonl_to_dir(prediction_dest_directory, file_name, predictions)

    evaluate_all_veracity_prediction(
        prediction_directory=prediction_dest_directory,
        predictions_file=file_name,
        split=split,
        ambifc_subset=ambifc_subset,
        overwrite=True,
        data_directory=data_directory
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
