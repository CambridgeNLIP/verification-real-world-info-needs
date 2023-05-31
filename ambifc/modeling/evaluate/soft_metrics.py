from collections import Counter
from itertools import permutations
from typing import List, Dict
import numpy as np
from scipy.stats import entropy


def to_label_distribution(annotations: List[str], labels: List[str]) -> np.ndarray:
    """
    Get the distribution (normalized) for the annotations as vector. The label per dimension is defined by
    the provided list of labels.

    :param annotations: List of annotations.
    :param labels: Lit of all labels. This is used to identify the dimension in the target vector for each label.
    """

    counts: Dict[str, int] = Counter(annotations)
    return np.array([
        counts.get(lbl, 0) / len(annotations) for lbl in labels
    ])


def get_all_allowed_ranks(annotations: List[str], labels: List[str]) -> List[np.ndarray]:
    """
    Return all legitimate orders of predicted labels. Multiple are possible if two or more labels share the same
    probability.

    LIMITATION: This assumes 3 possible labels!

    :param annotations: List of annotations.
    :param labels: Lit of all labels.
    """
    human_distribution: np.ndarray = to_label_distribution(annotations, labels=labels)
    assert len(human_distribution) == 3

    ranks: np.ndarray = human_distribution.argsort()

    if len(set(human_distribution)) == 3:
        # Three distinct probabilities -> Only one possibility.
        return [ranks]
    elif len(set(human_distribution)) == 1:
        # All have equal possibility -> All permutations are valid.
        return list(permutations(ranks))
    else:
        if human_distribution[ranks[0]] == human_distribution[ranks[1]]:
            # Either the first and second rank are equal
            return [ranks, np.array([ranks[1], ranks[0], ranks[2]])]
        else:
            # Or the second and third. (First and third is impossible, because then all three would be equal)
            return [ranks, np.array([ranks[0], ranks[2], ranks[1]])]


def get_instance_entropy_calibration_error(
        human_distribution: np.ndarray,
        predicted_distribution: np.ndarray,
        absolute: bool = True
) -> float:
    """
    From Baan et al (2022)
    """
    assert predicted_distribution.shape == human_distribution.shape
    calibration_error = entropy(predicted_distribution) - entropy(human_distribution)
    if absolute:
        calibration_error = np.abs(calibration_error)
    return calibration_error


def get_distillation_calibration_error(
    human_distribution: np.ndarray,
    predicted_distribution: np.ndarray
) -> float:
    """
    From Baan et al (2022)
    """
    assert np.max(predicted_distribution) <= 1.0 and np.min(predicted_distribution) >= 0, f'{predicted_distribution}'
    assert np.max(human_distribution) <= 1.0 and np.min(human_distribution) >= 0, f'{human_distribution}'
    distillation_calibration_error: float = np.sum(np.abs(predicted_distribution - human_distribution)) / 2

    # correct because of floating point errors (ensure it is between 1 and 0)
    distillation_calibration_error = max([min([distillation_calibration_error, 1.0]), 0.0])
    return distillation_calibration_error


def is_veracity_rank_correct(
    acceptable_label_ranks: List[np.ndarray],
    predicted_distribution: np.ndarray
) -> int:
    """
    From Baan et al (2022), but allowing multiple ranks
    """
    predicted_ranking: np.ndarray = predicted_distribution.argsort()
    for ranking in acceptable_label_ranks:
        if np.all(ranking == predicted_ranking):
            return 1
    return 0
