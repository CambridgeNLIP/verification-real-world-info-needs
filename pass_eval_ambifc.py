"""
Evaluate the passage level prediction.

Usage:
    pass_eval_ambifc.py <directory> <predictions> <split> <ambifc_subset> [--overwrite]
"""
import os
from os.path import join
from typing import Optional, List, Dict, Tuple

from docopt import docopt

from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.modeling.evaluate.eval import get_full_veracity_evaluation
from ambifc.util.fileutil import read_jsonl_from_dir, write_jsonl, write_json

DEFAULT_PATH_DATA_DIR: str = './data'


def data_to_dict(samples: List[Dict], field_claim_id: str, field_passage_id: str) -> Dict[Tuple[int, str], Dict]:
    result: Dict[Tuple[int, str], Dict] = dict()
    for sample in samples:
        key: Tuple[int, str] = (sample[field_claim_id], sample[field_passage_id])
        assert key not in result
        result[key] = sample
    return result


def evaluate_all_veracity_prediction(
        prediction_directory: str,
        predictions_file: str,
        split: str,
        ambifc_subset: str,
        overwrite: bool = False,
        data_directory: Optional[str] = DEFAULT_PATH_DATA_DIR
):
    print('Evaluate', prediction_directory, predictions_file)
    print('On', split, ambifc_subset)

    # At least evaluate on the full dataset as otherwise the subsets may not be correct.
    assert ambifc_subset in {TrainDataConfig.SUBSET_ALL_AMBIFC, TrainDataConfig.SUBSET_UNCERTAIN_ONLY_ALL}

    gold_data: Dict[Tuple[int, str], Dict] = data_to_dict(
        get_samples_for_ambifc_subset(
            ambifc_subset=ambifc_subset,
            split=split,
            data_directory=data_directory
        ),
        'claim_id',
        'wiki_passage'
    )

    keys_certain: List[Tuple[int, str]] = list(
        filter(lambda x: gold_data[x]['category'] == 'certain', gold_data.keys())
    )
    keys_uncertain: List[Tuple[int, str]] = list(
        filter(
            lambda x: gold_data[x]['category'] != 'certain' and len(gold_data[x]['passage_annotations']) >= 5,
            gold_data.keys()
        )
    )
    keys_uncertain_all: List[Tuple[int, str]] = list(
        filter(
            lambda x: gold_data[x]['category'] != 'certain', gold_data.keys()
        )
    )
    keys_certain_5plus_annotations: List[Tuple[int, str]] = list(
        filter(lambda x: len(gold_data[x]['passage_annotations']) >= 5, keys_certain)
    )

    keys_all_use: List[Tuple[int, str]] = keys_uncertain + keys_certain

    if ambifc_subset == TrainDataConfig.SUBSET_ALL_AMBIFC:
        assert set(keys_all_use) == set(gold_data.keys())

    # Get separate evaluations for uncertain / certain samples
    keys_to_evaluate: List[Tuple[str, List[Tuple[int, str]]]] = [
       ('certain', keys_certain),
       ('certain-5plus', keys_certain_5plus_annotations),
       ('uncertain', keys_uncertain),
       ('all-used', keys_all_use)
    ]

    if ambifc_subset == TrainDataConfig.SUBSET_UNCERTAIN_ONLY_ALL:
        keys_to_evaluate.append(('uncertain-all', keys_uncertain_all))

    # Only keep categories that actually exist
    keys_to_evaluate = list(
        filter(lambda x: len(x[1]) > 0, keys_to_evaluate)
    )
    for name, keys in keys_to_evaluate:
        print(f'{name}: evaluate {len(keys)} samples.')

    # Get predictions
    predicted_data: Dict[Tuple[int, str], Dict] = data_to_dict(
        list(read_jsonl_from_dir(prediction_directory, predictions_file)),
        'claim_id',
        'passage'
    )

    all_metrics: Dict = {}
    for name, keys in keys_to_evaluate:
        metrics: Dict = get_full_veracity_evaluation(
            gold_samples={key: gold_data[key] for key in keys},
            predicted_samples={key: predicted_data[key] for key in keys}
        )
        all_metrics[name] = metrics

    metrics_file_name: str = 'evaluation-' + predictions_file.replace('.jsonl', '.json')
    dest_path: str = join(prediction_directory, metrics_file_name)
    if overwrite and os.path.exists(dest_path):
        os.remove(dest_path)

    write_json(dest_path, all_metrics, pretty=True)


def main(args) -> None:
    directory: str = args['<directory>']
    predictions_file: str = args['<predictions>']
    split: str = args['<split>']
    ambifc_subset: str = args['<ambifc_subset>']
    overwrite: bool = args['--overwrite']

    assert os.path.exists(directory)
    assert os.path.exists(join(directory, predictions_file))
    assert split in ['train', 'dev', 'test']

    evaluate_all_veracity_prediction(
        directory, predictions_file, split, ambifc_subset, overwrite=overwrite
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
