"""
Select a threshold on which the evidence should be considered as evidence.

Usage:
    select-evidence-threshold.py <config> <subset> [--overwrite] [--local]
"""

from os.path import join
from typing import List, Dict, Tuple

import numpy as np
from docopt import docopt

from evidence_eval_ambifc import get_gold_sentence_annotations_as_dict, data_to_dict, evaluate_all_evidence_prediction
from ambifc.modeling.conf.config import Config
from ambifc.modeling.conf.labels import get_stance_label2int
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.evaluate.eval import get_full_evidence_evaluation
from ambifc.util.fileutil import read_json, read_jsonl, write_jsonl

DEFAULT_PATH_DATA_DIR: str = './data'


def assign_binary_evidence_label(output_type: str, prediction: Dict, threshold: float) -> str:
    if output_type == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
        evidence_probability: float = prediction['predicted_confidence']
    elif output_type == ModelConfig.OUTPUT_DISTRIBUTION:
        label2int: Dict[str, int] = get_stance_label2int()
        distribution: List[float] = prediction['predicted_distribution']
        assert len(distribution) == len(label2int)
        evidence_probability: float = sum([distribution[label2int[lbl]] for lbl in ['supporting', 'refuting']])
    else:
        raise NotImplementedError(output_type)

    if evidence_probability >= threshold:
        return 'evidence'
    else:
        return 'neutral'


def select_single_digit_evidence_threshold(
        predictions: Dict[Tuple[int, str, str], Dict],
        gold_data: Dict[Tuple[int, str, str], Dict],
        threshold_min: float,
        threshold_max: float,
        threshold_step: float,
        output_type: str
) -> float:

    assert output_type in {
        ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY, ModelConfig.OUTPUT_DISTRIBUTION
    }

    best_performance: float = -1
    best_threshold: float = -1
    keys: List = list(gold_data.keys())

    for threshold in np.arange(threshold_min, threshold_max + threshold_step, threshold_step):
        threshold = round(threshold, 2)

        # First re-assign the evidence labels based on the new threshold.
        for key in keys:
            current_prediction: Dict = predictions[key]
            evidence_label: str = assign_binary_evidence_label(output_type, current_prediction, threshold)
            current_prediction['predicted'] = evidence_label

        # Then evaluate
        metrics: Dict = get_full_evidence_evaluation(
            gold_samples={key: gold_data[key] for key in keys},
            predicted_samples={key: predictions[key] for key in keys}
        )
        evidence_f1: float = metrics["binary"]["evidence"]["f1-score"]
        if evidence_f1 > best_performance:
            best_performance = evidence_f1
            best_threshold = threshold

    assert best_threshold != -1
    return best_threshold


def main(args) -> None:
    config: Config = Config(read_json(args['<config>']))
    if args['--local']:
        prediction_directory: str = './sent_pred'
        model_directory: str = config.model_config.get_model_dest()
        prediction_directory = join(prediction_directory, model_directory)
    else:
        prediction_directory: str = config.get_prediction_directory()

    ambifc_subset: str = args['<subset>']
    file_name_test_dest: str = f'predictions.test.ambifc.evidence.jsonl'
    file_name_dev_dest: str = f'predictions.dev.ambifc.evidence.jsonl'
    assert config.model_config.get_model_task_type() == ModelConfig.TYPE_EVIDENCE

    if config.model_config.get_output_type() in {
        ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY, ModelConfig.OUTPUT_DISTRIBUTION
    }:
        file_name_dev_source: str = f'predictions.dev.ambifc.evidence.raw.jsonl'
        file_name_test_source: str = f'predictions.test.ambifc.evidence.raw.jsonl'

        prediction_file_tuning: str = join(prediction_directory, file_name_dev_source)
        predicted_data: Dict[Tuple[int, str, str], Dict] = data_to_dict(
            list(read_jsonl(prediction_file_tuning)),
            'claim_id',
            'passage',
            'sentence_key'
        )

        gold_data: Dict[Tuple[int, str, str], Dict] = get_gold_sentence_annotations_as_dict(
            DEFAULT_PATH_DATA_DIR, ambifc_subset, 'dev'
        )

        threshold_params: Dict = config.model_config.get_confidence_evidence_params()['evidence_threshold']
        min_t: float = threshold_params['min']
        max_t: float = threshold_params['max']
        step_t: float = threshold_params['step']

        threshold: float = select_single_digit_evidence_threshold(
            predicted_data,
            gold_data,
            threshold_min=min_t,
            threshold_max=max_t,
            threshold_step=step_t,
            output_type=config.model_config.get_output_type()
        )

        # Use selected threshold on TEST
        test_predictions: List[Dict] = list(
            read_jsonl(join(prediction_directory, file_name_test_source))
        )
        for sample in test_predictions:
            sample['evidence-threshold'] = threshold
            sample['predicted'] = assign_binary_evidence_label(
                config.model_config.get_output_type(), sample, threshold
            )

        write_jsonl(
            join(prediction_directory, file_name_test_dest),
            test_predictions
        )

        # Use selected threshold on DEV
        dev_predictions: List[Dict] = list(read_jsonl(prediction_file_tuning))
        for sample in dev_predictions:
            sample['evidence-threshold'] = threshold
            sample['predicted'] = assign_binary_evidence_label(
                config.model_config.get_output_type(), sample, threshold
            )

        write_jsonl(
            join(prediction_directory, file_name_dev_dest),
            dev_predictions
        )

    else:
        raise NotImplementedError(config.model_config.get_output_type())

    evaluate_all_evidence_prediction(
        prediction_directory=prediction_directory,
        predictions_file=file_name_test_dest,
        split='test',
        overwrite=args['--overwrite'],
        data_directory=DEFAULT_PATH_DATA_DIR
    )


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)
