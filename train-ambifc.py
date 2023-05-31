"""
Train models for AmbiFC. They will automatically get evaluated after training.

Usage:
    train-ambifc.py <config> [--overwrite]
"""
import torch

import os
import shutil
from os.path import join
from typing import Dict, List, Optional

from docopt import docopt
from transformers import AutoTokenizer, Trainer
from datasets import Dataset

from evidence_eval_ambifc import evaluate_all_evidence_prediction
from pass_eval_ambifc import evaluate_all_veracity_prediction
from predict_ambifc import predict
from ambifc.modeling.conf.config import Config
from ambifc.modeling.conf.labels import get_label2int
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.conf.train_config import TrainConfig
from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.modeling.dataset.get_dataset import get_dataset
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.modeling.distillation.distillation_trainer import DistillationTrainer
from ambifc.modeling.training.metrics_init import AmbiFCTrainMetrics, get_ambi_metrics_for_config
from ambifc.modeling.training.model_init import AmbiFCModelInit
from ambifc.modeling.training.training_arguments import get_training_args
from ambifc.util.fileutil import read_json


def move_checkpoint_to_model_directory(config: Config) -> None:
    """
    Moves the checkpoint to the model directory (avoid having an extra checkpoint directory with arbitrary name).
    """

    # Find the only existing checkpoint
    directory: str = config.model_config.get_model_dir()
    print('Looking in this directory:', directory)
    checkpoints: List[str] = os.listdir(directory)
    print('Found the following checkpoints (and expects 1)', checkpoints)
    assert len(checkpoints) == 1
    checkpoint_directory: str = join(directory, checkpoints[0])

    # Move the checkpoint
    for file in os.listdir(checkpoint_directory):
        shutil.move(join(checkpoint_directory, file), join(directory, file))

    shutil.rmtree(checkpoint_directory)


def train(config: Config, overwrite: bool) -> None:
    """
    Train a model based on the defined configuration.

    :param config: Contains all information of the model and used data.
    :param overwrite: Overwrite an existing already trained model.
    """

    assert torch.cuda.is_available()

    # Init Tokenizer
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(config.model_config.get_model_name())

    # Init Data (certain / uncertain / all AmbiFC)
    subset: str = config.train_data_config.get_ambifc_subset()
    data_dir: str = config.train_data_config.get_data_directory()

    # If training data should be reduced to a limited number of samples.
    max_num_training: Optional[int] = None
    max_num_shuffle_seed: Optional[int] = None

    if config.training_config.is_limit_number_training_samples():
        max_num_training, max_num_shuffle_seed = config.training_config.get_max_number_training_samples_params()

    # Get all raw training instances
    training_samples: List[Dict] = get_samples_for_ambifc_subset(
        subset, 'train', data_dir, max_num=max_num_training, max_num_shuffle_seed=max_num_shuffle_seed
    )
    print('Loaded', len(training_samples))
    print('CHECK ME')

    # Convert training instances into a dataset.
    dataset_train: Dataset = get_dataset(
        training_samples, config, tokenizer
    )

    # Get dev dataset
    dataset_dev: Dataset = get_dataset(
        get_samples_for_ambifc_subset(subset, 'dev', data_dir), config, tokenizer
    )

    # Init wrapper for model initialization.
    model_init = AmbiFCModelInit(
        model_name_or_path=config.model_config.get_model_name(),
        label2id=get_label2int(config.model_config),
        output_type=config.model_config.get_output_type(),
        set_to_eval=False
    )

    # Get the metrics needed during training - they differ based on task and subset.
    train_metrics: AmbiFCTrainMetrics = get_ambi_metrics_for_config(
        config.model_config, get_label2int(config.model_config)
    )

    # For distillation use a custom trainer that implements soft_cross_entropy loss.
    if config.model_config.get_output_type() == ModelConfig.OUTPUT_DISTRIBUTION:
        trainer: Trainer = DistillationTrainer(
            model_init=model_init.model_init,
            args=get_training_args(config, overwrite=overwrite),
            compute_metrics=train_metrics.compute_metrics,
            train_dataset=dataset_train,
            eval_dataset=dataset_dev,
            tokenizer=tokenizer
        )
    else:
        trainer: Trainer = Trainer(
            model_init=model_init.model_init,
            args=get_training_args(config, overwrite=overwrite),
            compute_metrics=train_metrics.compute_metrics,
            train_dataset=dataset_train,
            eval_dataset=dataset_dev,
            tokenizer=tokenizer
        )

    trainer.train()
    print('Done training. Move model checkpoint to top level directory ...')
    move_checkpoint_to_model_directory(config)
    print('Done.')


def main(args: Dict):
    # Read config and do model training
    config: Config = Config(read_json(args['<config>']))
    overwrite: bool = args['--overwrite']
    train(config, overwrite)

    # Always predict post training. Soft label subset (AmbiFC) includes all used samples.
    for ambifc_subset in [TrainDataConfig.SUBSET_ALL_AMBIFC]:
        for split in ['dev', 'test']:

            # Evidence predictions (T1)
            if config.model_config.get_model_task_type() == ModelConfig.TYPE_EVIDENCE:
                predict(
                    config=config, split=split, overwrite=overwrite, ambifc_subset=ambifc_subset
                )

                pred_directory: str = join(config.get_prediction_directory())
                prediction_file: str = f'predictions.{split}.{ambifc_subset}.evidence.jsonl'

                # Special case: When predicting the evidence probability or evidence distribution, selected evidence
                # is only detected after selecting a threshold.
                if not config.model_config.get_output_type() in {
                    ModelConfig.OUTPUT_BINARY_EVIDENCE_PROBABILITY,
                    ModelConfig.OUTPUT_DISTRIBUTION
                }:
                    evaluate_all_evidence_prediction(
                        prediction_directory=pred_directory,
                        predictions_file=prediction_file,
                        split=split,
                        overwrite=overwrite
                    )
            # Veracity prediction (T2) - predictions are based on either full-text or oracle evidence.
            # Predictions based on selected evidence must be made separately.
            else:
                if config.training_config.get_evidence_sampling_strategy() == TrainConfig.EVIDENCE_SELECTION_FULL:
                    sentence_predictions_for_eval: str = 'fulltext'
                else:
                    sentence_predictions_for_eval: str = 'oracle'

                predict(
                    config=config, split=split, overwrite=overwrite,
                    sentence_predictions=sentence_predictions_for_eval,
                    ambifc_subset=ambifc_subset
                )

                pred_directory: str = join(config.get_prediction_directory())
                prediction_file: str = f'predictions.{split}.{ambifc_subset}.veracity_{sentence_predictions_for_eval}-ev.jsonl'

                evaluate_all_veracity_prediction(
                    prediction_directory=pred_directory,
                    predictions_file=prediction_file,
                    split=split,
                    ambifc_subset=ambifc_subset,
                    overwrite=overwrite
                )


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)

