import os
from os.path import join
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import Dataset
from torch.nn import Softmax, Sigmoid
from transformers import Trainer

from ambifc.modeling.conf.config import Config
from ambifc.modeling.conf.labels import get_full_neutral_distribution
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.prediction.make_multi_label_predictions import make_multi_label_predictions_from_distribution
from ambifc.util.fileutil import write_jsonl


def to_predicted_single_labels(prediction_logits: torch.FloatTensor, int2label: Dict[int, str]) -> List[str]:
    """
    Convert the predicted logits of all samples within a batch into a list of single label strings.

    :param prediction_logits: Matrix of predicted logits.
    :param int2label: Dictionary that maps each dimension to a label name.
    """

    assert len(prediction_logits.size()) == 2, 'Logits must be of the size batch-size x labels'
    assert prediction_logits.size()[-1] == len(int2label.keys()), 'Logits must be of the size batch-size x labels'
    int_labels: List[int] = prediction_logits.argmax(dim=1).tolist()
    return list(map(lambda x: int2label[x], int_labels))


def to_probabilities_and_confidence(prediction_logits: torch.FloatTensor) -> Tuple[List[List[float]], List[float]]:
    """
    Extract the predicted probabilities and confidence from the predicted logits. The predicted probabilities are
    computed via Softmax, the confidence is the maximum probability.

    :param prediction_logits: Matrix of predicted logits.
    """
    assert len(prediction_logits.size()) == 2, 'Logits must be of the size batch-size x labels'
    soft_max = Softmax(dim=1)
    probabilities: torch.FloatTensor = soft_max(prediction_logits)
    confidence: torch.FloatTensor = probabilities.max(dim=1)[0]
    return probabilities.tolist(), confidence.tolist()


def to_predicted_multi_labels(
        prediction_probabilities: np.ndarray,
        int2label: Dict[int, str],
        threshold: float = 0.5
) -> List[List[str]]:
    """
    Convert a matrix of all logits from multiple samples into a list of multi-label predictions.

    :param prediction_probabilities: The probabilities of each individual class for each sample as a matrix.
    :param int2label: Dictionary to map each dimension to a label.
    :param threshold: Minimum probability to consider a class as predicted.
    """

    def to_labels(probabilities: np.ndarray) -> List[str]:
        assert probabilities.shape == (3,)
        class_indices: List[int] = np.where(probabilities >= threshold)[0].tolist()
        return list(map(lambda x: int2label[x], class_indices))

    num_samples, num_classes = prediction_probabilities.shape
    assert num_classes == 3
    return [
        to_labels(prediction_probabilities[i, :]) for i in range(num_samples)
    ]


def make_predictions_from_model_probability_distribution(
        predicted_logits: np.ndarray,
        dataset: Dataset,
        predicted_labels: List[str],
        predicted_probabilities: List[List[float]],
        predicted_confidences: List[float],
        int2label: Dict[int, str],
        is_veracity_prediction: bool,
        gold_label_is_distribution: bool
) -> List[Dict]:
    """
    Create predictions from predicted probability distributions such as single-label classification or annotation
    distillation.

    :param predicted_logits: matrix of all predicted logits from all samples of the dataset.
    :param dataset: Full dataset.
    :param predicted_labels: Predicted single labels of the label with the maximum probability.
    :param predicted_probabilities: Predicted probabilites for each sample.
    :param predicted_confidences: Predicted confidence values for each sample.
    :param int2label: Dictionary to map each dimension to a label.
    :param is_veracity_prediction: Boolean to indicate whether this is veracity prediction or not.
    :param gold_label_is_distribution: Boolean to indicate if the target during training was the annotation distribution.
    """

    # Keep predicted instances here.
    output: List[Dict] = []

    # Go over each sample.
    for i in range(len(predicted_logits)):

        # Get sample at the current position together with all respective prediction outputs.
        sample: Dict = dataset[i]
        logits: List[float] = predicted_logits[i]
        predicted_label: str = predicted_labels[i]
        predicted_probability: List[float] = predicted_probabilities[i]
        predicted_confidence: float = predicted_confidences[i]

        # Keep all fields for the output except for te following fields:
        current_pred = {
            k: sample[k] for k in sample if k not in ['input_ids', 'input_mask', 'attention_mask', 'token_type_ids']
        }

        # The field "label" may contain different values depending on the target. Check this here to get
        # uniform prediction files and separate a categorical label from the distribution.
        if gold_label_is_distribution:
            current_pred['target_distribution'] = current_pred['label']
            current_pred['label'] = np.argmax(current_pred['label'])

        # Include all fields into the output prediction sample.
        current_pred['label'] = int2label[current_pred['label']]
        current_pred['logits'] = logits
        current_pred['predicted_distribution'] = predicted_probability
        current_pred['predicted_confidence'] = predicted_confidence

        if is_veracity_prediction:
            current_pred['predicted'] = predicted_label

            # Set veracity prediction to neutral if no evidence was provided, but keep the original prediction..
            for key in ['logits', 'predicted_distribution', 'predicted_confidence', 'predicted']:
                current_pred[f'model_orig_{key}'] = current_pred[key]

            # If no sentences have been used, the predicton defaults to neutral.
            num_used_sentences: int = len(current_pred['sentence_keys'])
            if num_used_sentences == 0:
                current_pred['logits'] = get_full_neutral_distribution(int2label)
                current_pred['predicted_distribution'] = get_full_neutral_distribution(int2label)
                current_pred['predicted_confidence'] = 1.0
                current_pred['predicted'] = 'neutral'

                # Mark that this is NOT based on the veracity model, but because no evidence was selected.
                current_pred['is_evidence_based_prediction'] = False
            else:
                current_pred['is_evidence_based_prediction'] = True

            current_pred['multi_predicted'] = make_multi_label_predictions_from_distribution(
                int2label, current_pred['predicted_distribution']
            )

        elif not is_veracity_prediction and not gold_label_is_distribution:
            current_pred['predicted'] = predicted_label

        output.append(current_pred)
    return output


def store_predictions_from_probabilities(
        trainer: Trainer,
        dataset: Dataset,
        metric_key_prefix: str,
        int2label: Dict[int, str],
        is_veracity_prediction: bool,
        dest_name_predictions: str,
        gold_label_is_distribution: bool = False
):
    """
    Make and store predictions to a file.

    :param trainer: Trainer used to make inference.
    :param dataset: Full dataset.
    :param metric_key_prefix: TBA
    :param int2label: Dictionary to map each dimension to a label.
    :param is_veracity_prediction: Boolean to indicate whether this is veracity prediction or not.
    :param dest_name_predictions: Name of the prediction file
    :param gold_label_is_distribution: Boolean to indicate if the taret during training was the annotation distribution.

    """

    # Make inference
    predicted_logits, labels, metrics = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)

    predicted_logits = torch.FloatTensor(predicted_logits)
    predicted_labels: List[str] = to_predicted_single_labels(predicted_logits, int2label)
    predicted_probabilites, predicted_confidences = to_probabilities_and_confidence(predicted_logits)

    predicted_logits = predicted_logits.tolist()
    assert len(predicted_logits) == len(predicted_probabilites)
    assert len(predicted_logits) == len(predicted_confidences)

    # Create output predictions
    output: List[Dict] = make_predictions_from_model_probability_distribution(
        predicted_logits=predicted_logits,
        dataset=dataset,
        predicted_labels=predicted_labels,
        predicted_probabilities=predicted_probabilites,
        predicted_confidences=predicted_confidences,
        int2label=int2label,
        is_veracity_prediction=is_veracity_prediction,
        gold_label_is_distribution=gold_label_is_distribution
    )

    # if the task is evidence selection and the target is a distribution, the binary evidence labels mut later be found
    # via thresholding.
    if not is_veracity_prediction and gold_label_is_distribution:
        dest_name_predictions = dest_name_predictions.replace('.jsonl', '.raw.jsonl')

    # Write out all files.
    write_jsonl(dest_name_predictions, output)


def make_predictions_multi_label_classification(
    predicted_logits: np.ndarray,
    dataset: Dataset,
    predicted_labels: List[List[str]],
    predicted_probabilities: np.ndarray,
    predicted_distributions: List[List[float]],
    predicted_confidences: List[float],
    int2label: Dict[int, str],
    labels: np.ndarray,
    is_veracity_prediction: bool
) -> List[Dict]:
    """
        Create predictions from predicted probability distributions such as single-label classification or annotation
        distillation.

        :param predicted_logits: matrix of all predicted logits from all samples of the dataset.
        :param dataset: Full dataset.
        :param predicted_labels: Predicted single labels of the label with the maximum probability.
        :param predicted_probabilities: Predicted probabilites for each sample.
        :param predicted_distributions: Predicted distribution.
        :param predicted_confidences: Predicted confidence values for each sample.
        :param int2label: Dictionary to map each dimension to a label.
        :param labels: multi-label labels
        :param is_veracity_prediction: Boolean to indicate whether this is veracity prediction or not.
        """
    output: List[Dict] = []

    # This is okay because labels are always stored as 1 or 0
    gold_labels: List[List[str]] = to_predicted_multi_labels(labels, int2label, threshold=0.5)

    # Go over all samples
    for i in range(len(predicted_logits)):
        sample: Dict = dataset[i]
        logits: List[float] = predicted_logits[i]
        predicted_lbls: List[str] = predicted_labels[i]
        predicted_probability: List[float] = list(map(float, predicted_probabilities[i]))
        predicted_distribution: List[float] = list(map(float, predicted_distributions[i]))
        predicted_confidence: float = predicted_confidences[i]

        # The confidence is the averaged confidence for each class. Use the inverse for unpredicted classes,
        # i.e. if the model predicts 0% for class A, the model's confidence is 1-0 (an not 0).
        multi_predicted_confidence: float = float(np.mean([
            probability if probability >= 0.5 else 1 - probability
            for probability in predicted_probability
        ]))

        current_pred = {
            k: sample[k] for k in sample if k not in ['input_ids', 'input_mask', 'attention_mask', 'token_type_ids']
        }
        current_pred['multi_label'] = gold_labels[i]
        current_pred['logits'] = logits
        current_pred['predicted_distribution'] = predicted_distribution
        current_pred['predicted_confidence'] = predicted_confidence
        current_pred['multi_predicted_probabilities'] = predicted_probability
        current_pred['multi_predicted'] = predicted_lbls
        current_pred['multi_predicted_confidence'] = multi_predicted_confidence

        # Set veracity prediction to neutral if no evidence was provided.
        if is_veracity_prediction:
            for key in [
                'logits', 'predicted_distribution', 'predicted_confidence', 'multi_predicted',
                'multi_predicted_confidence', 'multi_predicted_probabilities'
            ]:
                current_pred[f'model_orig_{key}'] = current_pred[key]

            num_used_sentences: int = len(current_pred['sentence_keys'])
            if num_used_sentences == 0:
                current_pred['logits'] = get_full_neutral_distribution(int2label)
                current_pred['predicted_distribution'] = get_full_neutral_distribution(int2label)
                current_pred['predicted_confidence'] = 1.0
                current_pred['multi_predicted'] = ['neutral']
                current_pred['multi_predicted_probabilities'] = get_full_neutral_distribution(int2label)
                current_pred['multi_predicted_confidence'] = 1.0
                current_pred['is_evidence_based_prediction'] = False
            else:
                current_pred['is_evidence_based_prediction'] = True
        else:
            raise NotImplementedError()

        output.append(current_pred)
    return output


def store_predictions_multi_label_classification(
        trainer: Trainer,
        dataset: Dataset,
        metric_key_prefix: str,
        int2label: Dict[int, str],
        is_veracity_prediction: bool,
        dest_name_predictions: str
):
    """
    Make and store predictions for multi-label classification.
    """
    predicted_logits, labels, metrics = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)

    sigmoid: Sigmoid = Sigmoid()
    predicted_probabilities: np.ndarray = sigmoid(torch.FloatTensor(predicted_logits)).numpy()
    predicted_labels: List[List[str]] = to_predicted_multi_labels(predicted_probabilities, int2label, threshold=0.5)

    # We also interpret the logits as probability distribution to adhere to all metrics.
    predicted_distributions, predicted_confidences = to_probabilities_and_confidence(
        torch.FloatTensor(predicted_logits)
    )

    predicted_logits = predicted_logits.tolist()

    assert len(predicted_logits) == len(predicted_probabilities)
    assert len(predicted_logits) == len(predicted_distributions)
    assert len(predicted_logits) == len(predicted_labels)

    output: List[Dict] = make_predictions_multi_label_classification(
        predicted_logits=predicted_logits,
        dataset=dataset,
        predicted_labels=predicted_labels,
        predicted_probabilities=predicted_probabilities,
        predicted_distributions=predicted_distributions,
        predicted_confidences=predicted_confidences,
        int2label=int2label,
        labels=labels,
        is_veracity_prediction=is_veracity_prediction
    )

    write_jsonl(dest_name_predictions, output)


def store_predictions_for_evidence_probability(
    trainer: Trainer,
    dataset: Dataset,
    metric_key_prefix: str,
    is_veracity_prediction: bool,
    dest_name_predictions: str
):
    """
    Store the predictions of the evidence selection via binary evidence probability regression.
    """
    assert not is_veracity_prediction
    predicted_confidences, target_confidences, metrics = trainer.predict(dataset, metric_key_prefix=metric_key_prefix)
    predicted_confidences = predicted_confidences.reshape(-1)
    target_confidences = target_confidences.reshape(-1)
    assert predicted_confidences.shape == target_confidences.shape
    assert len(predicted_confidences) == len(dataset)

    output: List[Dict] = []
    # Go over all predictions
    for i in range(len(predicted_confidences)):
        sample: Dict = dataset[i]
        logits: List[float] = [float(predicted_confidences[i])]
        predicted_probability: List[float] = [float(predicted_confidences[i])]
        predicted_confidence: float = float(predicted_confidences[i])

        current_pred = {
            k: sample[k] for k in sample if k not in ['input_ids', 'input_mask', 'attention_mask', 'token_type_ids']
        }

        current_pred['logits'] = logits
        current_pred['predicted_distribution'] = predicted_probability
        current_pred['predicted_confidence'] = predicted_confidence
        current_pred['target_confidence'] = float(target_confidences[i])

        output.append(current_pred)

    write_jsonl(dest_name_predictions.replace('.jsonl', '.raw.jsonl'), output)


def store_predictions(
        config: Config,
        trainer: Trainer,
        dataset: Dataset,
        dataset_save_name: str,
        prediction_directory: str,
        metric_key_prefix: str,
        int2label: Dict[int, str],
        is_veracity_prediction: bool,
        overwrite: bool = False,
):
    """
    Store pedictions (and metrics) in the respective directoryas files.
    :param config:
        The experiment config of the trained model.
    :param trainer:
        The initialized trainer with the trained model to make inference
    :param dataset:
        The dataset including the samples to make inference on.
    :param dataset_save_name:
        Name of the dataset. This will be part of the resulting file names.
    :param prediction_directory:
        Files will be stored in this directory.
    :param metric_key_prefix:
        Metrics will have this prefix in the json file.
    :param int2label:
        converts integer values to string labels
    :param is_veracity_prediction:
        indicate whether it is veracity prediction or not. If set to true, and no evidence is provided, the prediction
        will be set to "neutral" by default.
    :param overwrite:
    :return:
    """
    if not os.path.exists(prediction_directory):
        os.makedirs(prediction_directory)

    # Prepare files
    dest_name_predictions: str = join(prediction_directory, f'predictions.{dataset_save_name}')
    if overwrite:
        if os.path.exists(dest_name_predictions):
            os.remove(dest_name_predictions)

    assert not os.path.exists(dest_name_predictions)

    if config.model_config.get_output_type() in {
        ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION, ModelConfig.OUTPUT_DISTRIBUTION
    }:
        if config.model_config.get_output_type() == ModelConfig.OUTPUT_DISTRIBUTION:
            gold_label_is_distribution: bool = True
        else:
            gold_label_is_distribution: bool = False

        store_predictions_from_probabilities(
            trainer=trainer,
            dataset=dataset,
            metric_key_prefix=metric_key_prefix,
            int2label=int2label,
            is_veracity_prediction=is_veracity_prediction,
            dest_name_predictions=dest_name_predictions,
            gold_label_is_distribution=gold_label_is_distribution
        )
    elif config.model_config.get_output_type() == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
        store_predictions_for_evidence_probability(
            trainer=trainer,
            dataset=dataset,
            metric_key_prefix=metric_key_prefix,
            is_veracity_prediction=is_veracity_prediction,
            dest_name_predictions=dest_name_predictions,
        )
    else:
        assert config.model_config.get_output_type() == ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION
        store_predictions_multi_label_classification(
            trainer=trainer,
            dataset=dataset,
            metric_key_prefix=metric_key_prefix,
            int2label=int2label,
            is_veracity_prediction=is_veracity_prediction,
            dest_name_predictions=dest_name_predictions,
        )
