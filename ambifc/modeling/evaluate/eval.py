from typing import Dict, Tuple, Set

from ambifc.modeling.evaluate.metrics import prfa_veracity_aggregated, prf_evidence_min1_from_samples, \
    compute_evidence_weighted_aggregated_veracity_score, compute_soft_veracity_metrics, \
    prf_evidence_from_sentence_predictions, get_instance_level_evidence_f1_dict
from ambifc.util.label_util import sentence_annotations_to_stance, sentence_annotations_to_binary, \
    make_prediction_binary


def get_full_veracity_evaluation(
        gold_samples: Dict[Tuple[int, str], Dict],
        predicted_samples: Dict[Tuple[int, str], Dict]
) -> Dict:
    """
    Get the veracity evaluation for the desired gold samples.
    :param gold_samples: Dictionary of all gold samples. Keys are claim_id and passage_id.
    :param predicted_samples: Dictionary of all predictions. Keys are claim_id and passage_id.
    """

    # Figure out if single-label predictions exist:
    one_sample: Dict = predicted_samples[list(predicted_samples.keys())[0]]
    if 'predicted' in one_sample.keys():
        has_single_labels_predictions: bool = True
    else:
        assert 'multi_predicted' in one_sample.keys()
        has_single_labels_predictions: bool = False

    # Compute metrics for evidence selection (T1)
    binary_evidence_metrics: Dict = prf_evidence_min1_from_samples(gold_samples, predicted_samples)

    # Get elementwise evidence F1 to compute join metrics (T1+T2)
    elm_wise_evidence_f1: Dict[Tuple[int, str], float] = get_instance_level_evidence_f1_dict(
        gold_samples, predicted_samples, apply_correction=False
    )

    # The same with correction, i.e. removing all selected evidence sentences if "neutral" is predicted as veracity.
    elm_wise_evidence_f1_corrected: Dict[Tuple[int, str], float] = get_instance_level_evidence_f1_dict(
        gold_samples, predicted_samples, apply_correction=True
    )

    # Compute metrics for aggregated labels (if single label predictions exist)
    if has_single_labels_predictions:
        agg_veracity_metrics: Dict = prfa_veracity_aggregated(gold_samples, predicted_samples)
        ev_weighted_aggregated_veracity_metrics: Dict = compute_evidence_weighted_aggregated_veracity_score(
            gold_samples, predicted_samples,
            elementwise_evidence_f1=elm_wise_evidence_f1,
            elementwise_evidence_f1_corrected=elm_wise_evidence_f1_corrected
        )
    else:
        agg_veracity_metrics: Dict = {}
        ev_weighted_aggregated_veracity_metrics: Dict = {}

    # Always compute metrics for soft labels.
    soft_veracity_metrics: Dict = compute_soft_veracity_metrics(
        gold_samples, predicted_samples,
        elementwise_evidence_f1=elm_wise_evidence_f1,
        elementwise_evidence_f1_corrected=elm_wise_evidence_f1_corrected
    )

    metrics: Dict = {
        'agg_veracity_metrics': agg_veracity_metrics,
        'evidence_weighted_aggregated_veracity_metrics': ev_weighted_aggregated_veracity_metrics,
        'soft_veracity_metrics': soft_veracity_metrics,
        'binary_evidence_metrics': binary_evidence_metrics,
    }

    return metrics


def get_full_evidence_evaluation(
    gold_samples: Dict[Tuple[int, str, str], Dict],
    predicted_samples: Dict[Tuple[int, str, str], Dict]
):
    """
    Get the evaluation for evidence selection T1.

    :param gold_samples: Dictionary of gold samples identified by keys of claim_id, passage_id, sentence_key.
    :param predicted_samples: Dictionary of sentence predictions identified by
    keys of claim_id, passage_id, sentence_key.
    """

    # Convert gold data to binary evidence labels
    id_to_binary_gold: Dict = {
        key: sentence_annotations_to_binary(gold_samples[key]['sentence_annotations'])
        for key in gold_samples.keys()
    }

    # Convert prediction to binary evidence predictions.
    id_to_binary_pred: Dict = {
        key: make_prediction_binary(predicted_samples[key]['predicted'])
        for key in gold_samples.keys()  # Gold key on purpose: All gold samples must be included.
    }

    # Compute binary evidence metrics (used as main metric)
    metrics: Dict = {
        'binary': prf_evidence_from_sentence_predictions(id_to_binary_gold, id_to_binary_pred)
    }

    # Test if stance labels exist.
    predictions: Set[str] = set(map(lambda x: predicted_samples[x]['predicted'], predicted_samples.keys()))

    # Only do evaluation based on stance labels if stance labels exist.
    if 'supporting' in predictions or 'refuting' in predictions:
        id_to_stance_gold: Dict = {
            key: sentence_annotations_to_stance(gold_samples[key]['sentence_annotations'])
            for key in gold_samples.keys()
        }

        id_to_stance_pred: Dict = {
            key: predicted_samples[key]['predicted']
            for key in gold_samples.keys()
        }
        metrics['stance'] = prf_evidence_from_sentence_predictions(id_to_stance_gold, id_to_stance_pred)
    else:
        if predictions == {'neutral'}:
            print('Warning: Could not determine automatically if evidence prediction is stance/binary.')

    return metrics
