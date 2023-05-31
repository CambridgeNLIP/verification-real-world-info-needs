from typing import Dict, Any, List, Tuple, Optional

import numpy as np
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm

from ambifc.modeling.conf.labels import make_label_list, get_stance_label2int
from ambifc.modeling.evaluate.soft_metrics import get_instance_entropy_calibration_error, to_label_distribution, \
    get_distillation_calibration_error, is_veracity_rank_correct, get_all_allowed_ranks
from ambifc.modeling.prediction.sentence_predictions import get_evidence_sentence_list, get_non_empty_sentence_keys


def get_instance_accuracy(gold_label: str, predicted_label: str) -> float:
    """
    Instance accuracy is 1.0 if correct, 0.0 otherwise.

    :param gold_label: Gold label.
    :param predicted_label: Predicted label.
    """
    if gold_label == predicted_label:
        return 1.0
    else:
        return 0.0


def get_instance_evidence_f1(
        gold_sentence_keys: List[str],
        predicted_sentence_key: List[str],
        all_non_empty_sentence_key: List[str]
) -> float:
    """
    Instance level evidence F1 score.
    :param gold_sentence_keys: All sentence keys that must be selected as evidence.
    :param predicted_sentence_key: All sentence keys that have been selected as evidence.
    :param all_non_empty_sentence_key: All sentences (excluding empty strings).
    """

    gold_labels: List[str] = []
    predicted_labels: List[str] = []
    for key in all_non_empty_sentence_key:
        gold_label: str = 'evidence' if key in gold_sentence_keys else 'neutral'
        pred_label: str = 'evidence' if key in predicted_sentence_key else 'neutral'
        gold_labels.append(gold_label)
        predicted_labels.append(pred_label)

    assert len(predicted_labels) == len(all_non_empty_sentence_key)

    # zero division=1 because it is correct to select none if none exists
    return f1_score(gold_labels, predicted_labels, pos_label='evidence', zero_division=1)


def get_instance_evidence_f1_from_sample(
        gold_sample: Dict,
        pred_sample: Dict,
        predicted_sentence_keys: Optional[List[str]] = None
) -> float:
    """
    Get the evidence F! score for a given sample. If the predicted evidence sentences must be overwritten (as needed
    for the correction rule), they can be provided separately.

    :param gold_sample: Gold sample.
    :param pred_sample: Prediction.
    :param predicted_sentence_keys: Sentence prediction that override the sentence prediction of the <pred_sample>.
    """
    all_sentences: List[str] = get_non_empty_sentence_keys(gold_sample)
    gold_sentences: List[str] = get_evidence_sentence_list(gold_sample)

    if predicted_sentence_keys is None:
        predicted_sentence_keys: List[str] = pred_sample['sentence_keys']

    return get_instance_evidence_f1(gold_sentences, predicted_sentence_keys, all_sentences)


def prf_evidence_min1_from_samples(gold: Dict[Tuple[int, str], Dict], pred: Dict[Tuple[int, str], Dict]) -> Dict:
    gold_labels: List[str] = []
    predicted_labels: List[str] = []

    keys: List[Any] = list(gold.keys())
    for key in keys:
        gold_sample: Dict = gold[key]
        pred_sample: Dict = pred[key]

        all_sentences: List[str] = get_non_empty_sentence_keys(gold_sample)
        gold_sentences: List[str] = get_evidence_sentence_list(gold_sample)
        predicted_sentences: List[str] = pred_sample['sentence_keys']

        for sentence_key in all_sentences:
            gold_label_sent: str = 'evidence' if sentence_key in gold_sentences else 'neutral'
            pred_label_sent: str = 'evidence' if sentence_key in predicted_sentences else 'neutral'
            gold_labels.append(gold_label_sent)
            predicted_labels.append(pred_label_sent)

    return classification_report(gold_labels, predicted_labels, zero_division=0, output_dict=True, digits=4)


def prf_evidence_from_sentence_predictions(
        gold: Dict[Tuple[int, str, str], str],
        pred: Dict[Tuple[int, str, str], str],
) -> Dict:
    """
    Compute the classification report for all evidence sentence predictions.

    :param gold: Dictionary that maps each sentence identified via claim_id, passage_id, sentence_key to a gold label.
    :param pred: Dictionary that maps each sentence identified via claim_id, passage_id, sentence_key to a predicted
    label.
    """
    evaluation_keys: List[Tuple[int, str, str]] = list(gold.keys())

    predicted_labels: List[str] = []
    gold_labels: List[str] = []

    for key in evaluation_keys:
        gold_labels.append(gold[key])
        predicted_labels.append(pred[key])

    return classification_report(gold_labels, predicted_labels, zero_division=0, output_dict=True, digits=4)


def prfa_veracity_aggregated(gold: Dict[Tuple[int, str], Dict], pred: Dict[Tuple[int, str], Dict]) -> Dict:
    """
    Compute Precision / Recall / F1 / Accuracy measures for the passage-level aggregated labels.

    :param gold: A dictionary containing all gold samples
    :param pred: A dictionary containing all predictions
    """
    keys: List[Tuple[int, str]] = list(gold.keys())
    gold_labels: List[str] = [gold[k]['labels']['passage'] for k in keys]
    pred_labels: List[str] = [pred[k]['predicted'] for k in keys]

    return classification_report(gold_labels, pred_labels, zero_division=0, output_dict=True, digits=4)


def compute_evidence_weighted_aggregated_veracity_score(
        gold: Dict[Tuple[int, str], Dict],
        pred: Dict[Tuple[int, str], Dict],
        elementwise_evidence_f1: Dict[Tuple[int, str], float],
        elementwise_evidence_f1_corrected: Dict[Tuple[int, str], float],
) -> Dict:
    """
    Compute the evidence-weighted metrics for the single-label (aggregated) evaluation.

    This multiplies the evidence F1 score (binary) with the accuracy on an instance level. The final performance is
    averaged over all instances.

    It computes two versions of the metric:
    * as-is: Use all predicted evidence sentences
    * corrected: If the model predicts neutral, all selected evidence sentences are removed.

    :param elementwise_evidence_f1_corrected:
    :param elementwise_evidence_f1:
    :param gold: A dictionary containing all gold samples
    :param pred: A dictionary containing all predictions
    """

    accuracies_passages: List[float] = []
    f1_scores_evidence: List[float] = []
    f1_scores_corrected_evidence: List[float] = []

    keys: List[Any] = list(gold.keys())

    for key in keys:
        gold_sample: Dict = gold[key]
        pred_sample: Dict = pred[key]

        gold_passage_label: str = gold_sample['labels']['passage']
        predicted_passage_label: str = pred_sample['predicted']

        accuracies_passages.append(get_instance_accuracy(gold_passage_label, predicted_passage_label))
        f1_scores_evidence.append(elementwise_evidence_f1[key])
        f1_scores_corrected_evidence.append(elementwise_evidence_f1_corrected[key])

    return {
        'ev_weighted_accuracy': np.mean(np.array(accuracies_passages) * np.array(f1_scores_evidence)),
        'ev_weighted_accuracy_corrected': np.mean(
            np.array(accuracies_passages) * np.array(f1_scores_corrected_evidence)
        )
    }


def get_instance_level_evidence_f1_dict(
        gold_samples: Dict[Tuple[int, str], Dict],
        predicted_samples: Dict[Tuple[int, str], Dict],
        apply_correction: bool = False
) -> Dict[Tuple[int, str], float]:
    """
    Create a dictionary that maps each claim/passage prediction to an instance-level evidence F1 score.

    :param gold_samples: Dictionary of gold samples identified via claim_id, passage_id.
    :param predicted_samples: Dictionary of predictions identified via claim_id, passage_id.
    :param apply_correction: If set to true, all selected evidence is removed when "neutral" is predicted as veracity.
    """
    result: Dict[Tuple[int, str], float] = dict()

    # Always iterate over gold samples, ensuring that predictions for all expected samples exist.
    for key in gold_samples.keys():
        gold_sample: Dict = gold_samples[key]
        pred_sample: Dict = predicted_samples[key]

        if 'predicted' in pred_sample:
            predicted_passage_label: str = pred_sample['predicted']
        else:
            predicted_passage_label: List[str] = pred_sample['multi_predicted']

        # Override selected evidence if correction rule is applied and only neutral is predicted.
        if apply_correction and (predicted_passage_label == 'neutral' or predicted_passage_label == ['neutral']):
            element_wise_evidence_f1: float = get_instance_evidence_f1_from_sample(
                gold_sample, pred_sample, predicted_sentence_keys=[]
            )
        else:
            element_wise_evidence_f1: float = get_instance_evidence_f1_from_sample(
                gold_sample, pred_sample
            )
        assert key not in result
        result[key] = element_wise_evidence_f1
    return result


def to_multi_label_matrix(target_labels: List[List[str]], label_names: List[str]) -> np.ndarray:
    """
    Create a multilabel matrix required for the classification report over multi-label classification.
    :param target_labels: list of all gold multi-labels for all instances.
    :param label_names: List of the label names required for ordering.
    """
    def map_multi_label_line(line_labels: List[str]) -> List[int]:
        return [1 if label in line_labels else 0 for label in label_names]

    return np.array(list(map(map_multi_label_line, target_labels)))


def compute_distillation_calibration_score(
    gold: Dict[Tuple[int, str], Dict],
    pred: Dict[Tuple[int, str], Dict],
) -> float:
    """
    Compute the averaged instance level calibration score based on 1- calibration error from Baan et al. (2022)
    """

    # Remember instance-level distillation calibration scores
    distillation_calibration_scores: List[float] = []

    # Iterate over all expected gold predictions
    label_order: List[str] = make_label_list(get_stance_label2int())
    for key in gold.keys():
        pred_sample: Dict = pred[key]
        gold_sample: Dict = gold[key]

        # Get gold and predicted distribution
        passage_annotations: List[str] = list(map(lambda x: x['label'], gold_sample['passage_annotations']))
        human_distribution: np.ndarray = to_label_distribution(passage_annotations, labels=label_order)
        predicted_distribution: np.ndarray = np.array(pred_sample['predicted_distribution'])
        assert predicted_distribution.shape == human_distribution.shape

        # compute the score
        distillation_calibration_error: float = get_distillation_calibration_error(
            human_distribution=human_distribution, predicted_distribution=predicted_distribution
        )
        distillation_calibration_scores.append(1 - distillation_calibration_error)

    return float(np.mean(distillation_calibration_scores))


def compute_soft_veracity_metrics(
    gold: Dict[Tuple[int, str], Dict],
    pred: Dict[Tuple[int, str], Dict],
    elementwise_evidence_f1: Dict[Tuple[int, str], float],
    elementwise_evidence_f1_corrected: Dict[Tuple[int, str], float],
) -> Dict:
    """
    Compute soft-label veracity metrics:
    :param elementwise_evidence_f1_corrected:
    :param elementwise_evidence_f1:
    :param gold: A dictionary containing all gold samples
    :param pred: A dictionary containing all predictions
    :return:
    """

    # Remember all instance-level metrics based on Baan et al. (2022)
    entropy_calibration_errors: List[float] = []
    distillation_calibration_errors: List[float] = []
    distillation_calibration_scores: List[float] = []
    rank_calibration_scores: List[int] = []

    # Remember all instance-level multi-label scores (precision/recall/f1/acc)
    multi_label_elementwise_p: List[float] = []
    multi_label_elementwise_r: List[float] = []
    multi_label_elementwise_f1: List[float] = []
    multi_label_elementwise_acc: Dict[Tuple[int, str], int] = dict()

    # Instance-level evidence F1 for joint metric (T1+T2) computation
    element_wise_evidence_f1_scores: List[float] = []
    element_wise_evidence_f1_scores_corrected: List[float] = []

    # Prepare for multi-label evaluation
    key_to_multi_labels_gold: Dict[Tuple[int, str], List[str]] = dict()
    label_list: List[str] = make_label_list(get_stance_label2int())

    # Iterate over all expected gold samples
    keys: List[Any] = list(gold.keys())
    for key in tqdm(keys):
        pred_sample: Dict = pred[key]
        gold_sample: Dict = gold[key]

        # Get gold labels for multi-label classification interpretation
        assert key not in key_to_multi_labels_gold
        key_to_multi_labels_gold[key] = list(set(map(lambda x: x['label'], gold_sample['passage_annotations'])))

        # Get evidence f1 for evidence based weighting (T1+T2)
        element_wise_evidence_f1_scores.append(elementwise_evidence_f1[key])
        element_wise_evidence_f1_scores_corrected.append(elementwise_evidence_f1_corrected[key])

        # Get target human veracity label distribution
        passage_annotations: List[str] = list(map(lambda x: x['label'], gold_sample['passage_annotations']))
        human_distribution: np.ndarray = to_label_distribution(passage_annotations, labels=label_list)

        # Get pedicted distributon
        predicted_distribution: np.ndarray = np.array(pred_sample['predicted_distribution'])
        assert predicted_distribution.shape == human_distribution.shape

        # Soft labels from Baan et al (2022)
        entropy_calibration_errors.append(get_instance_entropy_calibration_error(
            human_distribution=human_distribution, predicted_distribution=predicted_distribution
        ))

        distillation_calibration_error: float = get_distillation_calibration_error(
            human_distribution=human_distribution, predicted_distribution=predicted_distribution
        )
        distillation_calibration_errors.append(distillation_calibration_error)
        distillation_calibration_scores.append(1 - distillation_calibration_error)
        assert 0 <= distillation_calibration_error <= 1

        acceptable_label_ranks: List[np.ndarray] = get_all_allowed_ranks(passage_annotations, label_list)
        rank_calibration_scores.append(is_veracity_rank_correct(
            acceptable_label_ranks=acceptable_label_ranks, predicted_distribution=predicted_distribution
        ))

        # Multi-label classification evaluation
        multi_gold_labels: List[str] = key_to_multi_labels_gold[key]
        multi_pred_labels: List[str] = pred_sample['multi_predicted']
        scores: Dict = classification_report(
            to_multi_label_matrix([multi_gold_labels], label_list),
            to_multi_label_matrix([multi_pred_labels], label_list),
            target_names=label_list, output_dict=True, zero_division=1
        )
        multi_label_elementwise_p.append(scores['micro avg']['precision'])
        multi_label_elementwise_r.append(scores['micro avg']['recall'])
        multi_label_elementwise_f1.append(scores['micro avg']['f1-score'])
        assert key not in multi_label_elementwise_acc
        multi_label_elementwise_acc[key] = 1 if set(multi_gold_labels) == set(multi_pred_labels) else 0

    veracity_only_metrics = {
        'entropy_calibration_error': float(np.mean(entropy_calibration_errors)),
        'distillation_calibration_error': float(np.mean(distillation_calibration_errors)),
        'distillation_calibration_score': float(np.mean(distillation_calibration_scores)),
        'rank_calibration_score': np.mean(rank_calibration_scores),
        'multi_label_prf': classification_report(
            to_multi_label_matrix([key_to_multi_labels_gold[key] for key in keys], label_list),
            to_multi_label_matrix([pred[key]['multi_predicted'] for key in keys], label_list),
            target_names=label_list, output_dict=True
        ),
        'multi_label_accuracy': float(np.sum([multi_label_elementwise_acc[key] for key in keys]) / len(keys))
    }

    # Compute 1/2/3 label accuracy etc
    for num_gold_labels in range(1, 4):
        metric_name: str = f'{num_gold_labels}-label'
        use_keys: List[Tuple[int, str]] = list(filter(lambda k: len(key_to_multi_labels_gold[k]) == num_gold_labels, keys))
        veracity_only_metrics[metric_name] = {
            'support': len(use_keys),
            'accuracy': float(np.sum([multi_label_elementwise_acc[key] for key in use_keys]) / len(use_keys))
        }

    # Just to double check
    sample_averaged_multi_p: float = float(np.mean(multi_label_elementwise_p))
    sample_averaged_multi_r: float = float(np.mean(multi_label_elementwise_r))
    sample_averaged_multi_f1: float = float(np.mean(multi_label_elementwise_f1))
    assert round(sample_averaged_multi_p, 5) == round(
        veracity_only_metrics['multi_label_prf']['samples avg']['precision'], 5
    )
    assert round(sample_averaged_multi_r, 5) == round(
        veracity_only_metrics['multi_label_prf']['samples avg']['recall'], 5
    )
    assert round(sample_averaged_multi_f1, 5) == round(
        veracity_only_metrics['multi_label_prf']['samples avg']['f1-score'], 5
    )

    result: Dict = {
        'veracity-only': veracity_only_metrics,
        'evidence-weighted-veracity': {},
        'evidence-weighted-veracity-corrected': {}
    }

    # Add evidence-weighted joint metrics (T1+T2)
    rank_calibration_scores: np.ndarray = np.array(rank_calibration_scores)
    distillation_calibration_scores: np.ndarray = np.array(distillation_calibration_scores)
    for evidence_weighted_variant, ev_f1_scores in [
        ('evidence-weighted-veracity', element_wise_evidence_f1_scores),
        ('evidence-weighted-veracity-corrected', element_wise_evidence_f1_scores_corrected)
    ]:
        evidence_scores: np.ndarray = np.array(ev_f1_scores)
        assert len(evidence_scores) == len(rank_calibration_scores)
        assert len(evidence_scores) == len(distillation_calibration_scores)

        result[evidence_weighted_variant]['rank_calibration_score'] = float(
            np.mean(rank_calibration_scores * evidence_scores)
        )
        result[evidence_weighted_variant]['distillation_calibration_score'] = float(
            np.mean(distillation_calibration_scores * evidence_scores)
        )

        result[evidence_weighted_variant]['multi_label_avg_precision'] = float(
            np.mean(sample_averaged_multi_p * evidence_scores)
        )

        result[evidence_weighted_variant]['multi_label_avg_recall'] = float(
            np.mean(sample_averaged_multi_r * evidence_scores)
        )

        result[evidence_weighted_variant]['multi_label_avg_f1'] = float(
            np.mean(sample_averaged_multi_f1 * evidence_scores)
        )

        result[evidence_weighted_variant]['multi_label_accuracy'] = float(
            np.mean(np.array([multi_label_elementwise_acc[key] for key in keys]) * evidence_scores)
        )

    return result
