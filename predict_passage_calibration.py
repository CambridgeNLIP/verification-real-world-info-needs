"""
Applies temperature scaling as post-hoc calibration method.

Usage:
    predict_passage_calibration.py temperature-scaling <model-directory> <tuning-data> <test-data> <temperature-min> <temperature-max> <temperature-step> <ambifc_subset>
"""
import pathlib
from copy import copy
from os.path import join
from typing import Dict, List, Tuple, Iterable

import numpy as np
import torch.nn
from docopt import docopt

from pass_eval_ambifc import data_to_dict, evaluate_all_veracity_prediction
from ambifc.modeling.conf.labels import make_int2label, get_stance_label2int
from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.modeling.evaluate.metrics import compute_distillation_calibration_score
from ambifc.modeling.prediction.make_multi_label_predictions import make_multi_label_predictions_from_distribution
from ambifc.util.fileutil import write_jsonl_to_dir, read_jsonl

DEFAULT_CALIBRATION_PREDICTION_DIRECTORY: str = join(pathlib.Path(__file__).parent.resolve(), './veracity_calibration')
DEFAULT_CALIBRATION_EVALUATION_DIRECTORY: str = join(
    pathlib.Path(__file__).parent.resolve(), './veracity_calibration-evaluation'
)
DEFAULT_VERACITY_PREDICTION_DIRECTORY: str = join(pathlib.Path(__file__).parent.resolve(), './veracity_pred')
DEFAULT_DATA_DIRECTORY: str = join(pathlib.Path(__file__).parent.resolve(), './data')


def make_temperature_scaling(
    predictions: Dict[Tuple[int, str], Dict],
    temperature: float
) -> Dict[Tuple[int, str], Dict]:
    """
    Run temperature scaling with temperature over given predictions.
    """
    result: Dict[Tuple[int, str], Dict] = dict()
    softmax: torch.nn.Softmax = torch.nn.Softmax(dim=0)

    int2label: Dict[int, str] = make_int2label(get_stance_label2int())
    for key in predictions.keys():
        sample = predictions[key]

        # Do not rescale if the prediction defaults to neutral because of no selected evidence
        if not sample['is_evidence_based_prediction']:
            result[key] = copy(sample)
        else:
            scaled_sample: Dict = {
                key: copy(sample[key]) for key in sample.keys()
                if key not in ['logits', 'predicted_distribution', 'predicted_confidence']
            }

            # Temperature Scaling happens here
            new_logits: torch.FloatTensor = torch.FloatTensor(sample['logits']) / temperature
            new_predicted_distribution: torch.FloatTensor = softmax(new_logits)

            # Re-compute the outputs based on the rescaled logits.
            scaled_sample['logits'] = new_logits.tolist()
            scaled_sample['predicted_distribution'] = new_predicted_distribution.tolist()
            scaled_sample['logits'] = max(scaled_sample['predicted_distribution'])

            scaled_sample['multi_predicted'] = make_multi_label_predictions_from_distribution(
                int2label, scaled_sample['predicted_distribution']
            )

            result[key] = scaled_sample
    return result


def run_temperature_scaling_search(
        predicted_samples: List[Dict],
        ambifc_subset: str,
        min_t: float,
        max_t: float,
        step_t: float,
        data_directory: str
) -> Iterable[Tuple[float, float]]:
    """
    Search over all thresholds for temperature scaling.
    """

    # Load relevant gold samples
    gold_data: Dict[Tuple[int, str], Dict] = data_to_dict(
        get_samples_for_ambifc_subset(
            ambifc_subset=ambifc_subset,
            split='dev',
            data_directory=data_directory
        ), 'claim_id', 'wiki_passage'
    )

    predicted_data: Dict[Tuple[int, str], Dict] = data_to_dict(
        list(filter(lambda x: (x['claim_id'], x['passage']) in gold_data, predicted_samples)),
        'claim_id',
        'passage'
    )

    # Make sure all samples from the relevant subset (fom gold) have predictions. it is okay if
    # predictions include a superset of the relevant samples.
    assert set(gold_data.keys()) & set(predicted_data.keys()) == set(gold_data.keys())
    predicted_data = {
        k: predicted_data[k] for k in gold_data.keys()
    }
    print('Tuning based on', len(predicted_data.keys()), 'entries.')

    # Go over all possible temperature values
    for temperature in np.arange(min_t, max_t + step_t, step_t):
        temperature = round(temperature, 2)
        temperature_scaled_predictions: Dict[Tuple[int, str], Dict] = make_temperature_scaling(
            predicted_data, temperature
        )

        # Use distillation calibration score as metric
        dist_cs: float = compute_distillation_calibration_score(
            gold_data, temperature_scaled_predictions
        )
        print(f'Temperature: {temperature} -> DistCS: {dist_cs}')

        yield temperature, dist_cs


def main(args) -> None:
    model_directory: str = args['<model-directory>']
    tuning_data_name: str = args['<tuning-data>']
    test_data_name: str = args['<test-data>']

    min_temperature: float = float(args['<temperature-min>'])
    max_temperature: float = float(args['<temperature-max>'])
    step_temperature: float = float(args['<temperature-step>'])

    ambifc_subset: str = args['<ambifc_subset>']

    prediction_dest_directory: str = join(DEFAULT_CALIBRATION_PREDICTION_DIRECTORY, model_directory)

    original_predictions_tuning: List[Dict] = list(read_jsonl(
        join(DEFAULT_VERACITY_PREDICTION_DIRECTORY, join(model_directory, tuning_data_name))
    ))

    original_predictions_testing: Dict[Tuple[int, str], Dict] = data_to_dict(
        list(read_jsonl(
            join(DEFAULT_VERACITY_PREDICTION_DIRECTORY, join(model_directory, test_data_name))
        )),
        'claim_id',
        'passage'
    )

    if args['temperature-scaling']:
        temperatures_and_scores = run_temperature_scaling_search(
            predicted_samples=original_predictions_tuning,
            ambifc_subset=ambifc_subset,
            min_t=min_temperature,
            max_t=max_temperature,
            step_t=step_temperature,
            data_directory=DEFAULT_DATA_DIRECTORY
        )
        best_temperature, best_score = sorted(list(temperatures_and_scores), key=lambda x: x[-1])[-1]
        print('Using the best temperature of', best_score, 'reaching DistCS:', best_score, 'on the dev set.')

        scaled_testing_predictions: Dict[Tuple[int, str], Dict] = make_temperature_scaling(
            original_predictions_testing, best_temperature
        )

        file_name: str = f'temp-scaling-{str(best_temperature).replace(".", "-")}__{test_data_name}'

    else:
        raise NotImplementedError()

    # Write predictions
    write_jsonl_to_dir(prediction_dest_directory, file_name, [
        scaled_testing_predictions[key] for key in scaled_testing_predictions.keys()
    ])

    evaluate_all_veracity_prediction(
        prediction_directory=prediction_dest_directory,
        predictions_file=file_name,
        split='test',
        ambifc_subset=TrainDataConfig.SUBSET_ALL_AMBIFC,
        overwrite=True,
        data_directory=DEFAULT_DATA_DIRECTORY
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
