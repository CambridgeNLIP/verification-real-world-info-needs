from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error, r2_score
from torch.nn import Sigmoid, Softmax, CrossEntropyLoss

from ambifc.modeling.conf.labels import get_label2int, make_int2label, get_evidence_conversion_dict, \
    get_binary_evidence_label2int, make_label_list
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.distillation.distillation_trainer import soft_cross_entropy
from ambifc.modeling.evaluate.soft_metrics import get_instance_entropy_calibration_error, \
    get_distillation_calibration_error


class AmbiFCTrainMetrics:
    """
    Base class to compute relevant metrics during training based on the selected model type
    """

    def __init__(self, label2int: Dict[str, int]):
        self.label2int: Dict[str, int] = label2int
        self.int2label: Dict[int, str] = make_int2label(label2int)

    def compute_metrics(self, model_outputs) -> Dict:
        label_ids = model_outputs.label_ids
        predictions = model_outputs.predictions
        return self._compute_metrics_from_predictions_and_labels(predictions, label_ids)

    def _compute_metrics_from_predictions_and_labels(self, predictions, label_ids) -> Dict:
        """
        Override this metric
        """
        raise NotImplementedError()


class AmbiFCVeracitySingleLabelMetrics(AmbiFCTrainMetrics):
    """
    Metrics when training single-label classification for veracity
    """

    def _compute_metrics_from_predictions_and_labels(self, predictions, label_ids) -> Dict:

        prediction_ids = predictions.argmax(-1)
        metrics: Dict = classification_report(label_ids, prediction_ids, zero_division=0, output_dict=True)
        result: Dict = {
            'accuracy': metrics['accuracy']
        }

        prf: List[str] = 'precision recall f1-score'.split(' ')
        for metric in prf:
            result[f'macro-{metric}'] = metrics['macro avg'][metric]
            result[f'micro-{metric}'] = metrics['weighted avg'][metric]

        class_names: List[str] = [key for key in metrics if str(key) not in ['accuracy', 'macro avg', 'weighted avg']]
        for cls in class_names:
            cls_name: str = self.int2label[int(cls)]
            for key in metrics[cls]:
                result[f'{cls_name}-{key}'] = metrics[cls][key]

        return result


class AmbiFCEvidenceDistributionMetrics(AmbiFCTrainMetrics):
    """
    Metrics when training annotation distillation for evidence.
    """
    def __init__(self, label2int: Dict[str, int], model_config: ModelConfig):
        super().__init__(label2int)
        self.model_config: ModelConfig = model_config

    def _compute_metrics_from_predictions_and_labels(self, predictions: np.ndarray, target: np.ndarray) -> Dict:
        cross_entropy_loss: float = soft_cross_entropy(
            torch.FloatTensor(predictions), torch.FloatTensor(target)
        ).item()
        output: Dict = {
            'neg_cross_entropy': -float(cross_entropy_loss)
        }

        return output


class AmbiFCVeracityDistributionMetrics(AmbiFCTrainMetrics):

    """
    Metrics when training annotation distillation for veracity.
    """

    def __init__(self, label2int: Dict[str, int]):
        super().__init__(label2int)
        self.softmax: Softmax = Softmax(dim=1)

    def _compute_metrics_from_predictions_and_labels(
            self, predictions: np.ndarray, labels: np.ndarray
    ) -> Dict:
        cross_entropy_loss: float = soft_cross_entropy(
            torch.FloatTensor(predictions), torch.FloatTensor(labels)
        ).item()
        probabilities: torch.FloatTensor = self.softmax(torch.FloatTensor(predictions)).numpy()

        assert predictions.shape == labels.shape
        human_calibration_errors: List[float] = []
        human_distillation_scores: List[float] = []

        for i in range(len(probabilities)):
            predicted: np.ndarray = probabilities[i, :]
            gold: np.ndarray = labels[i, :]
            human_calibration_errors.append(get_instance_entropy_calibration_error(predicted, gold))
            human_distillation_scores.append(1 - get_distillation_calibration_error(predicted, gold))

        output: Dict = {
            'neg_cross_entropy': -float(cross_entropy_loss),
            'human_calibration_error': float(np.mean(human_calibration_errors)),
            'human_distillation_score': float(np.mean(human_distillation_scores))
        }

        return output


class AmbiFCVeracityMultiLabelMetrics(AmbiFCTrainMetrics):
    """
        Metrics when training multi-label classification for veracity.
    """

    def __init__(self, label2int: Dict[str, int], threshold: float = 0.5):
        super().__init__(label2int)
        self.sigmoid: Sigmoid = Sigmoid()
        self.label_list: List[str] = make_label_list(label2int)
        self.threshold: float = threshold

    def _compute_metrics_from_predictions_and_labels(
            self, predictions: np.ndarray, label_ids: np.ndarray
    ) -> Dict:
        predictions = np.copy(predictions)
        predictions = self.sigmoid(torch.FloatTensor(predictions)).numpy()
        filter_predicted_classes: np.ndarray = predictions >= self.threshold

        # Make one-hot predictions
        predictions[filter_predicted_classes] = 1.0
        predictions[~filter_predicted_classes] = 0.0

        metrics: Dict = classification_report(label_ids, predictions, target_names=self.label_list, output_dict=True)

        result_dict: Dict = {}
        for label in self.label_list:
            for metric_name in ['precision', 'recall', 'f1-score', 'support']:
                result_dict[f'{label}-{metric_name}'] = metrics[label][metric_name]

        for metric_name in ['precision', 'recall', 'f1-score', 'support']:
            result_dict[f'macro-{metric_name}'] = metrics['macro avg'][metric_name]
            result_dict[f'micro-{metric_name}'] = metrics['weighted avg'][metric_name]

        return result_dict


class AmbiFCEvidenceProbabilityRegressionMetrics(AmbiFCTrainMetrics):
    """
        Metrics when evidence probability regression.
    """
    def __init__(self, label2int: Dict[str, int], model_config: ModelConfig):
        super().__init__(label2int)
        self.model_config: ModelConfig = model_config
        self.evidence_label_variant: str = model_config.get_evidence_label_variant()

    def _compute_metrics_from_predictions_and_labels(self, predictions, target) -> Dict:
        target = target.reshape(-1, 1)
        assert predictions.shape == target.shape

        mse = mean_squared_error(target, predictions)
        mae = mean_absolute_error(target, predictions)
        r2 = r2_score(target, predictions)

        return {
            "mse": mse,
            "mae": mae,
            "r2": r2,
            "negative-mse": -mse
        }


class AmbiFCEvidenceSingleLabelMetrics(AmbiFCTrainMetrics):
    """
    Metrics when training single-label classification for veracity
    """

    def __init__(self, label2int: Dict[str, int], model_config: ModelConfig):
        super().__init__(label2int)
        self.model_config: ModelConfig = model_config
        self.evidence_label_variant: str = model_config.get_evidence_label_variant()

    def _compute_metrics_from_predictions_and_labels(self, predictions, label_ids) -> Dict:

        predicted_classes: List[str] = list(map(lambda x: self.int2label[x], predictions.argmax(-1)))
        gold_labels: List[str] = list(map(lambda x: self.int2label[x], label_ids))

        metrics: Dict = self.get_binary_evidence_metrics(predicted_classes, gold_labels)
        if self.evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_STANCE:
            stance_metrics: Dict = self.get_stance_metrics(predicted_classes, gold_labels)
            for key in stance_metrics:
                assert key not in metrics
                metrics[key] = stance_metrics[key]
        return metrics

    def get_binary_evidence_metrics(self, predicted_labels: List[str], gold_labels: List[str]) -> Dict:
        if self.evidence_label_variant == ModelConfig.EVIDENCE_VARIANT_STANCE:
            conversion_dict: Dict[str, str] = get_evidence_conversion_dict()
            predicted_labels = list(map(lambda x: conversion_dict[x], predicted_labels))
            gold_labels = list(map(lambda x: conversion_dict[x], gold_labels))

        metrics: Dict = classification_report(gold_labels, predicted_labels, zero_division=0, output_dict=True)
        result: Dict = {
            'accuracy': metrics['accuracy']
        }
        prf: List[str] = 'precision recall f1-score'.split(' ')
        for metric in prf:
            result[f'binary-macro-{metric}'] = metrics['macro avg'][metric]
            result[f'binary-micro-{metric}'] = metrics['weighted avg'][metric]

        print('metrics', metrics)
        for label in get_binary_evidence_label2int().keys():
            for key in metrics[label]:
                result[f'{label}-{key}'] = metrics[label][key]

        return result

    def get_stance_metrics(self, predicted_labels: List[str], gold_labels: List[str]) -> Dict:
        metrics: Dict = classification_report(gold_labels, predicted_labels, zero_division=0, output_dict=True)
        result: Dict = {
            'accuracy': metrics['accuracy']
        }
        prf: List[str] = 'precision recall f1-score'.split(' ')
        for metric in prf:
            result[f'macro-{metric}'] = metrics['macro avg'][metric]
            result[f'micro-{metric}'] = metrics['weighted avg'][metric]

        for label in self.label2int.keys():
            for key in metrics[label]:
                result[f'{label}-{key}'] = metrics[label][key]

        return {
            f'stance-{key}': result[key] for key in result
        }


def get_ambi_metrics_for_config(model_config: ModelConfig, label2int: Dict[str, int]) -> AmbiFCTrainMetrics:
    """
    Get the correct metric implementation based on the model configuration.
    """

    model_type: str = model_config.get_model_task_type()
    model_output_type: str = model_config.get_output_type()

    if model_type == ModelConfig.TYPE_VERACITY:
        if model_output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
            return AmbiFCVeracitySingleLabelMetrics(label2int)
        elif model_output_type == ModelConfig.OUTPUT_MULTI_LABEL_CLASSIFICATION:
            return AmbiFCVeracityMultiLabelMetrics(label2int)
        elif model_output_type == ModelConfig.OUTPUT_DISTRIBUTION:
            return AmbiFCVeracityDistributionMetrics(label2int)
        else:
            raise NotImplementedError()
    else:
        assert model_type == ModelConfig.TYPE_EVIDENCE
        if model_output_type == ModelConfig.OUTPUT_SINGLE_LABEL_CLASSIFICATION:
            return AmbiFCEvidenceSingleLabelMetrics(label2int, model_config)
        elif model_output_type == ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY:
            return AmbiFCEvidenceProbabilityRegressionMetrics(label2int, model_config)
        elif model_output_type == ModelConfig.OUTPUT_DISTRIBUTION:
            return AmbiFCEvidenceDistributionMetrics(label2int, model_config)
        else:
            raise NotImplementedError(model_output_type)
