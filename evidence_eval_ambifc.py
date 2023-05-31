"""
Evaluate the evidence prediction on the sentence level.

Usage:
    evidence_eval_ambifc.py <directory> <predictions> <split> [--overwrite]
"""
import os
from os.path import join
from typing import Optional, List, Dict, Tuple, Iterable

from docopt import docopt

from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.modeling.evaluate.eval import get_full_evidence_evaluation
from ambifc.modeling.prediction.sentence_predictions import get_non_empty_sentence_keys
from ambifc.util.fileutil import read_jsonl_from_dir, write_json

DEFAULT_PATH_DATA_DIR: str = './data'


def data_to_dict(
        samples: List[Dict],
        field_claim_id: str,
        field_passage_id: str,
        sentence_key: str
) -> Dict[Tuple[int, str, str], Dict]:
    result: Dict[Tuple[int, str, str], Dict] = dict()
    for sample in samples:
        key: Tuple[int, str, str] = (sample[field_claim_id], sample[field_passage_id], sample[sentence_key])
        assert key not in result
        result[key] = sample
    return result


def get_gold_sentence_annotations(data_directory: str, ambifc_subset: str, split: str) -> List[Dict]:
    return [
        {
            'claim_id': sample['claim_id'],
            'wiki_passage': sample['wiki_passage'],
            'sentence_key': sentence_key,
            'sentence_annotations': list(map(lambda x: x['annotation'], sample['sentence_annotations'][sentence_key])),
            'category': sample['category'],
            'passage_annotations': sample['passage_annotations']
        }
        for sample in get_samples_for_ambifc_subset(
            ambifc_subset=ambifc_subset,
            split=split,
            data_directory=data_directory
        )
        for sentence_key in get_non_empty_sentence_keys(sample)
    ]


def get_gold_sentence_annotations_as_dict(
        data_directory: str,
        ambifc_subset: str,
        split: str) -> Dict[Tuple[int, str, str], Dict]:
    sentence_annotations_gold: List[Dict] = get_gold_sentence_annotations(data_directory, ambifc_subset, split)
    return data_to_dict(
        sentence_annotations_gold,
        'claim_id',
        'wiki_passage',
        'sentence_key'
    )


def evaluate_all_evidence_prediction(
        prediction_directory: str,
        predictions_file: str,
        split: str,
        overwrite: bool = False,
        data_directory: Optional[str] = DEFAULT_PATH_DATA_DIR
):

    gold_data: Dict[Tuple[int, str, str], Dict] = get_gold_sentence_annotations_as_dict(
        data_directory, TrainDataConfig.SUBSET_ALL_AMBIFC, split
    )

    keys_certain: List[Tuple[int, str, str]] = list(
        filter(lambda x: gold_data[x]['category'] == 'certain', gold_data.keys())
    )
    keys_uncertain: List[Tuple[int, str, str]] = list(
        filter(lambda x: gold_data[x]['category'] != 'certain', gold_data.keys())
    )
    keys_certain_5plus_annotations: List[Tuple[int, str, str]] = list(
        filter(lambda x: len(gold_data[x]['passage_annotations']) >= 5, keys_certain)
    )

    # Get separate evaluations for uncertain / certain samples
    keys_to_evaluate: Iterable[Tuple[str, List[Tuple[int, str, str]]]] = [
        ('all', list(gold_data.keys())),
        ('certain', keys_certain),
        ('certain-5plus', keys_certain_5plus_annotations),
        ('uncertain', keys_uncertain)
    ]

    # Only keep categories that actually exist
    keys_to_evaluate = list(
        filter(lambda x: len(x[1]) > 0, keys_to_evaluate)
    )
    for name, keys in keys_to_evaluate:
        print(f'{name}: evaluate {len(keys)} samples.')

    # Get predictions
    predicted_data: Dict[Tuple[int, str, str], Dict] = data_to_dict(
        list(read_jsonl_from_dir(prediction_directory, predictions_file)),
        'claim_id',
        'passage',
        'sentence_key'
    )

    all_metrics: Dict = {}
    for name, keys in keys_to_evaluate:
        metrics: Dict = get_full_evidence_evaluation(
            gold_samples={key: gold_data[key] for key in keys},
            predicted_samples={key: predicted_data[key] for key in keys}
        )
        all_metrics[name] = metrics

    metrics_file_name: str = 'evaluation-' + predictions_file.replace('.jsonl', '.json')
    dest_path: str = join(prediction_directory, metrics_file_name)
    if overwrite and os.path.exists(dest_path):
        os.remove(dest_path)

    write_json(dest_path, all_metrics, pretty=True)
    print('Write: dest_path', dest_path)
    print('Done.')


def main(args) -> None:
    directory: str = args['<directory>']
    predictions_file: str = args['<predictions>']
    split: str = args['<split>']
    overwrite: bool = args['--overwrite']

    assert os.path.exists(directory)
    assert os.path.exists(join(directory, predictions_file))
    assert split in ['train', 'dev', 'test']

    evaluate_all_evidence_prediction(
        directory, predictions_file, split, overwrite=overwrite
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
