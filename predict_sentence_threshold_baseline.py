"""
Applies majority based veracity prediction using the sentence labels.

Usage:
    predict_sentence_threshold_baseline.py majority <prediction-directory> <sentence-prediction-file> <data_directory> <split> <ambifc_subset> [--threshold=<threshold>]
    predict_sentence_threshold_baseline.py last-sentence <prediction-directory> <sentence-prediction-file> <data_directory> <split> <ambifc_subset> [--threshold=<threshold>]
    predict_sentence_threshold_baseline.py most-confident <prediction-directory> <sentence-prediction-file> <data_directory> <split> <ambifc_subset> [--threshold=<threshold>]
"""
import os
from collections import defaultdict, Counter
from os.path import join
from typing import Optional, List, Dict, Tuple, Iterable, Set

from docopt import docopt

from pass_eval_ambifc import evaluate_all_veracity_prediction
from ambifc.modeling.conf.labels import get_full_neutral_distribution, get_stance_label2int, make_int2label
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.util.fileutil import read_jsonl, write_jsonl_to_dir
from ambifc.util.label_util import sentence_annotations_to_stance
import pathlib


DEFAULT_BASELINE_PREDICTION_DIRECTORY: str = join(pathlib.Path(__file__).parent.resolve(), './veracity_baselines')
DEFAULT_BASELINE_EVALUATION_DIRECTORY: str = join(
    pathlib.Path(__file__).parent.resolve(), './veracity_baselines-evaluation'
)


def get_last_sentence_prediction(sentence_predictions: List[Dict]) -> str:
    """
    Return the last prediction.
    """
    return sorted(sentence_predictions, key=lambda x: int(x['sentence_key']))[-1]['predicted']


def make_majority_prediction(sentence_predictions: List[Dict]) -> str:
    """
    Return the majority vote.
    """
    labels: List[str] = list(map(lambda x: x['predicted'], sentence_predictions))
    counter: Counter = Counter(labels)
    if counter.get('refuting', 0) == counter.get('supporting', 0):
        # as tiebreaker use last label
        predicted: str = get_last_sentence_prediction(sentence_predictions)
    else:
        predicted: str = counter.most_common(1)[0][0]
    return predicted


def make_most_confident_prediction(sentence_predictions: List[Dict]) -> str:
    """
    Consider the sentence of non-neutral labels with the highest confidence for the veracity label.
    Only consider sentences with a non-neutral confidence of >= threshold.
    :param sentence_predictions:
    :param threshold:
    :return:
    """
    return sorted(sentence_predictions, key=lambda x: x['predicted_confidence'])[-1]['predicted']


def make_single_baseline_prediction(
    sentence_predictions: List[Dict],
    baseline_variant: str,
    threshold: Optional[float] = None,
    remove_neutral: bool = True
):
    int2lbl: Dict[int, str] = make_int2label(get_stance_label2int())

    if remove_neutral:
        keep_sentences: Iterable[Dict] = filter(lambda s: s['predicted'] != 'neutral', sentence_predictions)
    else:
        keep_sentences: Iterable[Dict] = sentence_predictions

    if threshold is not None:
        keep_sentences: List[Dict] = list(filter(lambda s: s['predicted_confidence'] >= threshold, keep_sentences))
    else:
        keep_sentences: List[Dict] = list(keep_sentences)

    if len(keep_sentences) == 0:
        # If all sentences are neutral
        return {
            'sentence_keys': [],
            'logits': get_full_neutral_distribution(int2lbl),
            'predicted_distribution': get_full_neutral_distribution(int2lbl),
            'is_evidence_based_prediction': False,
            'predicted': 'neutral',
            'predicted_confidence': 1.0,
            'multi_predicted': ['neutral']
        }
    else:

        labels: List[str] = list(map(lambda x: x['predicted'], keep_sentences))
        assert 'evidence' not in labels, f'binary evidence selection model is not applicable!'

        if baseline_variant == 'most-confident':
            predicted: str = sorted(keep_sentences, key=lambda x: x['predicted_confidence'])[-1]['predicted']
        elif baseline_variant == 'last-sentence-prediction':
            predicted: str = get_last_sentence_prediction(keep_sentences)
        elif baseline_variant == 'majority':
            predicted: str = make_majority_prediction(keep_sentences)
        else:
            raise NotImplementedError(baseline_variant)

        # Adjust to the format of veracity predictions.
        counter: Counter = Counter(labels)
        logits: List[int] = [counter.get(int2lbl[i], 0) for i in range(len(int2lbl.keys()))]
        predicted_distribution: List[float] = [logit / sum(logits) for logit in logits]
        predicted_multi_label: List[str] = [
            int2lbl[i] for i in range(len(predicted_distribution)) if predicted_distribution[i] >= 0.2
        ]
        return {
            'sentence_keys': list(map(lambda x: x['sentence_key'], keep_sentences)),
            'logits': logits,
            'predicted_distribution': predicted_distribution,
            'predicted_confidence': max(predicted_distribution),
            'predicted': predicted,
            'is_evidence_based_prediction': True,
            'multi_predicted': predicted_multi_label
        }


def make_baseline_predictions(
    predictions: Dict[Tuple[int, str], List[Dict]],
    sentence_prediction_from: str,
    baseline_variant: str,
    baseline_params: Optional[Dict]
) -> List[Dict]:

    required_keys: Set[str] = {
        'sentence_keys', 'logits', 'predicted_distribution', 'is_evidence_based_prediction', 'predicted',
        'predicted_confidence', 'multi_predicted'
    }
    result: List[Dict] = []
    for claim_id, passage_id in predictions.keys():
        sentence_predictions: List[Dict] = predictions[(claim_id, passage_id)]

        if baseline_variant in {'majority', 'last-sentence-prediction', 'most-confident'}:
            threshold: float = baseline_params['threshold'] if 'threshold' in baseline_params else None
            baseline_passage_prediction: Dict = make_single_baseline_prediction(
                sentence_predictions=sentence_predictions,
                baseline_variant=baseline_variant,
                threshold=threshold
            )
        else:
            raise NotImplementedError(baseline_variant)

        if set(baseline_passage_prediction.keys()) & required_keys != required_keys:
            raise ValueError(f'Need these keys: {required_keys} but found these keys: {baseline_passage_prediction.keys()}')

        sample = {
            'claim_id': claim_id,
            'passage': passage_id,
            'entity_name': sentence_predictions[0]['entity_name'],
            'section_title': sentence_predictions[0]['section_title'],
            'claim': sentence_predictions[0]['claim'],
            'sentence_prediction_from': sentence_prediction_from,
        }
        for key in baseline_passage_prediction:
            sample[key] = baseline_passage_prediction[key]
        sample['evidence'] = list(
            map(
                lambda p: p['sentence'],
                filter(lambda p: p['sentence_key'] in sample['sentence_keys'], sentence_predictions)
            )
        )

        result.append(sample)
    return result


def get_oracle_sentence_predictions(
        data_directory: str, split: str, ambifc_subset: str
) -> Dict[Tuple[int, str], List[Dict]]:
    """
    Convert the gold labels into the prediction-style format.
    :param data_directory: Directory path including all AmbiFC files
    :param split: "train"/"dev"/"test"
    :param ambifc_subset: one of the data selection strategies.
    :return:
    """
    samples: List[Dict] = get_samples_for_ambifc_subset(ambifc_subset, split, data_directory)
    result: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    int2lbl: Dict[int, str] = make_int2label(get_stance_label2int())

    for oracle_sample in samples:
        claim_id: int = oracle_sample['claim_id']
        passage: str = oracle_sample['wiki_passage']
        entity_name: str = oracle_sample['entity']
        section_title: str = oracle_sample['section']
        claim: str = oracle_sample['claim']

        sample_id: Tuple[int, str] = (claim_id, passage)
        for sentence_key in oracle_sample['sentences'].keys():

            # Ignore empty sentences.
            if len(oracle_sample['sentences'][sentence_key].strip()) > 0:
                sentence: str = oracle_sample['sentences'][sentence_key].strip()
                sentence_annotations: List[str] = list(
                    map(lambda ann: ann['annotation'], oracle_sample['sentence_annotations'][sentence_key])
                )
                annotation_counts: Dict[str, int] = Counter(sentence_annotations)
                logits: List[int] = [annotation_counts.get(int2lbl[i], 0) for i in range(len(int2lbl.keys()))]
                predicted_distribution: List[float] = list(map(lambda val: val/sum(logits), logits))
                predicted_confidence: float = max(predicted_distribution)
                predicted: str = sentence_annotations_to_stance(sentence_annotations)

                result[sample_id].append({
                    'claim_id': claim_id,
                    'passage': passage,
                    'sentence_key': sentence_key,
                    'entity_name': entity_name,
                    'section_title': section_title,
                    'claim': claim,
                    'sentence': sentence,
                    'logits': logits,
                    'predicted_distribution': predicted_distribution,
                    'predicted_confidence': predicted_confidence,
                    'predicted': predicted
                })
    return result


def get_stance_sentence_predictions(sentence_prediction_file: str) -> Dict[Tuple[int, str], List[Dict]]:
    """
    Create a dictionary mapping all sentence predictions per claim_id, passage_id
    """

    # Aggregate all sentences together.
    result: Dict[Tuple[int, str], List[Dict]] = defaultdict(list)
    sentence_predictions: Iterable[Dict] = read_jsonl(sentence_prediction_file)
    for sentence_prediction in sentence_predictions:
        key: Tuple[int, str] = (sentence_prediction['claim_id'], sentence_prediction['passage'])
        result[key].append(sentence_prediction)
    return result


def main(args) -> None:
    prediction_dest_directory: str = join(DEFAULT_BASELINE_PREDICTION_DIRECTORY, args['<prediction-directory>'])
    evaluation_dest_directory: str = join(DEFAULT_BASELINE_EVALUATION_DIRECTORY, args['<prediction-directory>'])
    sent_predictions_file: str = args['<sentence-prediction-file>']
    ambifc_subset: str = args['<ambifc_subset>']
    split: str = args['<split>']
    data_directory: str = args['<data_directory>']

    if sent_predictions_file == 'oracle':
        evidence_file_appendix: str = f'{split}.{ambifc_subset}.oracle-ev'
        sentence_predictions: Dict[Tuple[int, str], List[Dict]] = get_oracle_sentence_predictions(
            data_directory=data_directory, split=split, ambifc_subset=ambifc_subset
        )
    else:
        _, evidence_file_appendix = os.path.split(sent_predictions_file)
        evidence_file_appendix = evidence_file_appendix.replace('.jsonl', '-ev')
        sentence_predictions: Dict[Tuple[int, str], List[Dict]] = get_stance_sentence_predictions(sent_predictions_file)

    print('Found', len(sentence_predictions), 'passages.')

    if not os.path.exists(prediction_dest_directory):
        os.makedirs(prediction_dest_directory)
    if not os.path.exists(evaluation_dest_directory):
        os.makedirs(evaluation_dest_directory)

    threshold: str = args['--threshold']
    if threshold is not None:
        threshold_appendix: str = f'-t{threshold.replace(".", "")}'
    else:
        threshold_appendix: str = ''

    if args['majority']:
        # Get majority vote of sentence level predictions (excluding neutral of course)

        file_name: str = f'majority-{evidence_file_appendix}{threshold_appendix}.jsonl'
        results: List[Dict] = make_baseline_predictions(
            predictions=sentence_predictions,
            sentence_prediction_from=sent_predictions_file,
            baseline_variant='majority',
            baseline_params={'threshold': float(threshold)} if threshold is not None else {}
        )
    elif args['last-sentence']:
        file_name: str = f'last-sentence-prediction-{evidence_file_appendix}{threshold_appendix}.jsonl'
        results: List[Dict] = make_baseline_predictions(
            predictions=sentence_predictions,
            sentence_prediction_from='sent_predictions_file todo',
            baseline_variant='last-sentence-prediction',
            baseline_params={'threshold': float(threshold)} if threshold is not None else {}
        )
    elif args['most-confident']:
        file_name: str = f'most-confident-sentence-{evidence_file_appendix}{threshold_appendix}.jsonl'
        results: List[Dict] = make_baseline_predictions(
            predictions=sentence_predictions,
            sentence_prediction_from='sent_predictions_file todo',
            baseline_variant='most-confident',
            baseline_params={'threshold': float(threshold)} if threshold is not None else {}
        )
    else:
        raise NotImplementedError()

    write_jsonl_to_dir(prediction_dest_directory, file_name, results)
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
