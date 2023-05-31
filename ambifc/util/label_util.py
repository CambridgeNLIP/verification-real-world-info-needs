from collections import Counter
from typing import List, Set, Tuple, Dict, Optional

import torch
from torch.nn import Softmax

ALLOWED_LABELS: Set[str] = {
    'supporting', 'refuting', 'neutral'
}


def make_prediction_binary(prediction: str) -> str:
    if prediction in {'supporting', 'refuting'}:
        return 'evidence'
    return prediction


def get_non_neutral_annotations(annotations: List[str]) -> List[str]:
    assert set(annotations) | ALLOWED_LABELS == ALLOWED_LABELS, f'Bad annotations: {annotations}'
    return list(filter(lambda x: x != 'neutral', annotations))


def sentence_annotations_to_stance(annotations: List[str], majority: str = 'supporting') -> str:
    non_neutral_annotations: List[str] = get_non_neutral_annotations(annotations)
    if len(non_neutral_annotations) == 0:
        return 'neutral'

    label_counts: List[Tuple[str, int]] = Counter(non_neutral_annotations).most_common()
    assert len(label_counts) < 3
    if len(label_counts) == 1:
        # Only one non-neutral label
        return label_counts[0][0]
    elif label_counts[0][1] == label_counts[1][1]:
        # If both are equal -> use majority (supporting)
        return majority
    else:
        return label_counts[0][0]


def sentence_annotations_to_binary(annotations: List[str]) -> str:
    non_neutral_annotations: List[str] = get_non_neutral_annotations(annotations)
    if len(non_neutral_annotations) > 0:
        return 'evidence'
    else:
        return 'neutral'


def sentence_annotation_to_binary_evidence_confidence(annotations: List[str]) -> float:
    annotations = list(map(make_prediction_binary, annotations))
    num_evidence = len(list(filter(lambda x: x == 'evidence', annotations)))
    return num_evidence / len(annotations)


def make_soft_label_probability_distribution(labels: List[str], int2lbl: Dict[int, str]) -> List[float]:
    assert len(labels) > 0
    counts: Counter = Counter(labels)
    return [
        counts.get(int2lbl[i], 0) / len(labels) for i in range(len(int2lbl.keys()))
    ]


def make_soft_label_softmax_distribution(
        labels: List[str], int2lbl: Dict[int, str], temperature: Optional[float] = None, normalize: bool = True
) -> List[float]:
    if temperature is None:
        temperature = 1.

    if normalize:
        distribution: torch.FloatTensor = torch.FloatTensor(make_soft_label_probability_distribution(labels, int2lbl))
    else:
        counts: Counter = Counter(labels)
        distribution: torch.FloatTensor = torch.FloatTensor([
            counts.get(int2lbl[i], 0) / len(labels) for i in range(len(int2lbl.keys()))
        ])
    softmax: Softmax = Softmax(dim=0)
    return softmax(distribution / temperature).tolist()
