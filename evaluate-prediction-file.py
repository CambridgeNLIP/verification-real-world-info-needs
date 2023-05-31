"""
Evaluate a prediction file.

Run like: evaluate.py ../predictions.json ../data dev

Usage:
    evaluate-prediction-file.py eval <path-to-predictions> <path-to-gold-data> <split>
    evaluate-prediction-file.py add-instance-metrics <path-to-predictions> <path-to-gold-data> <split>
    """

from os.path import join
from typing import Dict, Iterable, List, Tuple,  Set

import numpy as np
from docopt import docopt
from sklearn.metrics import classification_report
from tqdm import tqdm

from ambifc.modeling.conf.labels import make_label_list, get_stance_label2int
from ambifc.modeling.evaluate.eval import get_full_veracity_evaluation
from ambifc.modeling.evaluate.metrics import to_multi_label_matrix
from ambifc.modeling.evaluate.soft_metrics import to_label_distribution, get_instance_entropy_calibration_error, \
    get_distillation_calibration_error, is_veracity_rank_correct, get_all_allowed_ranks
from ambifc.util.fileutil import read_json, read_jsonl, write_json, write_jsonl


def get_prediction_data(filepath: str) -> Iterable[Dict]:
    if filepath.endswith('.json'):
        return read_json(filepath)
    else:
        assert filepath.endswith('.jsonl')
        return read_jsonl(filepath)


def evaluate(args: Dict) -> None:
    prediction_path: str = args['<path-to-predictions>']
    data_dir: str = args['<path-to-gold-data>']
    split: str = args['<split>']

    predictions: Dict[Tuple[int, str], Dict] = {
        (pred['claim_id'], pred['passage']): pred
        for pred in get_prediction_data(prediction_path)
    }

    print('Loaded', len(predictions), 'predictions.')

    gold_samples: Dict[Tuple[int, str], Dict] = {
        (entry['claim_id'], entry['wiki_passage']): entry
        for key in ['certain', 'uncertain']
        for entry in read_jsonl(join(data_dir, f'{split}.{key}.jsonl'))
        if (entry['claim_id'], entry['wiki_passage']) in predictions
    }

    # All predictions must have a gold equivalent.
    assert set(predictions.keys()) == set(gold_samples.keys()), f'{len(predictions)} VS {len(gold_samples)}'

    metrics: Dict = get_full_veracity_evaluation(gold_samples, predictions)
    write_json(prediction_path.replace('json', '-METRICS.json'), metrics, pretty=True)
    print(metrics)


def add_instance_metrics(args: Dict) -> None:
    prediction_path: str = args['<path-to-predictions>']
    data_dir: str = args['<path-to-gold-data>']
    split: str = args['<split>']

    predictions: List[Dict] = list(get_prediction_data(prediction_path))
    prediction_keys: Set = set(map(lambda x: (x['claim_id'], x['passage']), predictions))

    gold_samples: Dict[Tuple[int, str], Dict] = {
        (entry['claim_id'], entry['wiki_passage']): entry
        for key in ['certain', 'uncertain']
        for entry in read_jsonl(join(data_dir, f'{split}.{key}.jsonl'))
        if (entry['claim_id'], entry['wiki_passage']) in prediction_keys
    }

    assert prediction_keys == set(gold_samples.keys())
    label_order: List[str] = make_label_list(get_stance_label2int())

    for i, prediction in enumerate(tqdm(predictions)):
        gold_sample: Dict = gold_samples[(prediction['claim_id'], prediction['passage'])]

        passage_annotations: List[str] = list(map(lambda x: x['label'], gold_sample['passage_annotations']))
        human_distribution: np.ndarray = to_label_distribution(passage_annotations, labels=label_order)
        predicted_distribution: np.ndarray = np.array(prediction['predicted_distribution'])
        assert predicted_distribution.shape == human_distribution.shape

        # Entropy Calibration Error
        ent_ce: float = get_instance_entropy_calibration_error(
            human_distribution=human_distribution, predicted_distribution=predicted_distribution
        )

        # Distillation Calibration Score
        dist_cs: float = 1 - get_distillation_calibration_error(
            human_distribution=human_distribution, predicted_distribution=predicted_distribution
        )

        # Rank Calibration Score
        acceptable_label_ranks: List[np.ndarray] = get_all_allowed_ranks(passage_annotations, label_order)
        rank_cs: float = is_veracity_rank_correct(
            acceptable_label_ranks=acceptable_label_ranks, predicted_distribution=predicted_distribution
        )

        # F1
        multi_gold_labels: List[str] = list(set(map(lambda x: x['label'], gold_sample['passage_annotations'])))
        multi_pred_labels: List[str] = prediction['multi_predicted']
        scores: Dict = classification_report(
            to_multi_label_matrix([multi_gold_labels], label_order),
            to_multi_label_matrix([multi_pred_labels], label_order),
            target_names=label_order, output_dict=True, zero_division=1
        )
        multi_label_f1: float = scores['micro avg']['f1-score']

        prediction['instance-metrics'] = {
            'ent_ce': ent_ce,
            'dist_cs': dist_cs,
            'rank_cs': rank_cs,
            'multi_label_f1': multi_label_f1
        }

    write_jsonl(prediction_path, predictions)


def main(args: Dict) -> None:

    if args['eval']:
        evaluate(args)
    elif args['add-instance-metrics']:
        add_instance_metrics(args)


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
