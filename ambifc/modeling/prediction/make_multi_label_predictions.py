from typing import Dict, List


# For multilabel prediction: When the model only predict a probability distribution over all classes (as in single-label
# classification), any class with a probability of >= this number is considered predicted.
# We choose 0.2 as in most cases 5 annotators annotated a sample, and we consider each annotation (at least 20%)
MULTI_LABEL_PREDICTION_THRESHOLD_FROM_DISTRIBUTION: float = 0.2


def make_multi_label_predictions_from_distribution(
        int2label: Dict[int, str], probability_distribution: List[float]
) -> List[str]:
    return [
        int2label[i] for i in range(len(probability_distribution))
        if probability_distribution[i] >= MULTI_LABEL_PREDICTION_THRESHOLD_FROM_DISTRIBUTION
    ]
