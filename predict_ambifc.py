"""
General prediction script

Usage:
    predict_ambifc.py <config> <split> [--overwrite] [--subset=<ambifc-subset>] [--sentences=<sentence-predictions>] [--pred-file-appendix=<appendix>] [--eval]
"""
import os.path
from os.path import join
from typing import Dict, Tuple, List, Optional

from docopt import docopt
from transformers import AutoTokenizer, Trainer
from datasets import Dataset

from pass_eval_ambifc import evaluate_all_veracity_prediction
from ambifc.modeling.conf.config import Config
from ambifc.modeling.conf.labels import get_label2int, make_int2label
from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.modeling.dataset.get_dataset import get_dataset
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from ambifc.modeling.prediction.sentence_predictions import get_oracle_sentence_prediction_dict, \
    get_instance_to_predicted_evidence_dict, get_fulltext_sentence_prediction_dict
from ambifc.modeling.training.metrics_init import AmbiFCTrainMetrics, get_ambi_metrics_for_config
from ambifc.modeling.training.model_init import AmbiFCModelInit
from ambifc.modeling.prediction.store_predictions import store_predictions
from ambifc.util.fileutil import read_json, read_jsonl

SENTENCE_PREDICTION_DIRECTORY: str = "./sent_pred"


def predict(
        config: Config,
        split: str,
        overwrite: bool,
        sentence_predictions: Optional[str] = None,
        sentence_prediction_directory: str = SENTENCE_PREDICTION_DIRECTORY,
        pred_file_appendix: str = '',
        ambifc_subset: Optional[str] = None
) -> str:

    assert split in ['train', 'dev', 'test']

    if ambifc_subset is None:
        ambifc_subset = TrainDataConfig.SUBSET_ALL_AMBIFC

    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(config.model_config.get_model_name())
    data_dir: str = config.train_data_config.get_data_directory()
    ambifc_train_metrics: AmbiFCTrainMetrics = get_ambi_metrics_for_config(
        config.model_config, get_label2int(config.model_config)
    )
    model_init = AmbiFCModelInit(
        # Load stored model
        model_name_or_path=config.model_config.get_model_dir(),
        label2id=get_label2int(config.model_config),
        output_type=config.model_config.get_output_type(),
        set_to_eval=True
    )

    trainer: Trainer = Trainer(
        model_init=model_init.model_init,
        tokenizer=tokenizer,
        compute_metrics=ambifc_train_metrics.compute_metrics
    )

    samples: List[Dict] = get_samples_for_ambifc_subset(ambifc_subset, split, data_dir)

    # Sentence predictions are NOT required when predicting evidence sentences of course
    # However, when predicting the veracity, the sentences to rely on must be specified.
    if sentence_predictions is None:
        sentence_prediction_dict: Optional = None
        sentence_prediction_source: Optional = None
        save_name: str = f'{split}.{ambifc_subset}.{config.model_config.get_model_task_type()}.jsonl'
        assert config.model_config.get_model_task_type() == ModelConfig.TYPE_EVIDENCE
    elif sentence_predictions == 'fulltext':
        sentence_prediction_dict: Optional[Dict[Tuple[int, str], List[str]]] = get_fulltext_sentence_prediction_dict(
            samples
        )
        sentence_prediction_source: Optional[str] = 'fulltext'
        save_name: str = f'{split}.{ambifc_subset}.{config.model_config.get_model_task_type()}_fulltext-ev.jsonl'
    elif sentence_predictions == 'oracle':
        sentence_prediction_dict: Optional[Dict[Tuple[int, str], List[str]]] = get_oracle_sentence_prediction_dict(
            samples
        )
        sentence_prediction_source: Optional[str] = 'oracle'
        save_name: str = f'{split}.{ambifc_subset}.{config.model_config.get_model_task_type()}_oracle-ev.jsonl'
    else:
        sentence_prediction_file: str = join(sentence_prediction_directory, sentence_predictions)
        assert os.path.exists(sentence_prediction_file), f'Does not exist: "{sentence_prediction_file}"'
        sentence_prediction_source: Optional[str] = sentence_prediction_file
        sentence_prediction_dict: Optional[Dict[Tuple[int, str], List[str]]] = get_instance_to_predicted_evidence_dict(
            list(read_jsonl(sentence_prediction_file))
        )

        assert pred_file_appendix != '' and pred_file_appendix is not None
        save_name: str = f'{split}.{ambifc_subset}.{config.model_config.get_model_task_type()}_{pred_file_appendix}-ev.jsonl'

    dataset: Dataset = get_dataset(
        samples, config, tokenizer, sentence_prediction_dict, sentence_prediction_source
    )

    is_veracity_prediction: bool = (config.model_config.get_model_task_type() == 'veracity')
    store_predictions(
        config=config,
        trainer=trainer,
        dataset=dataset,
        dataset_save_name=save_name,
        prediction_directory=config.get_prediction_directory(),
        metric_key_prefix=split,
        int2label=make_int2label(get_label2int(config.model_config)),
        is_veracity_prediction=is_veracity_prediction,
        overwrite=overwrite
    )
    return f'predictions.{save_name}'


def main(args: Dict):
    config: Config = Config(read_json(args['<config>']))
    prediction_file: str = predict(
        config=config,
        split=args['<split>'],
        overwrite=args['--overwrite'],
        sentence_predictions=args['--sentences'],
        pred_file_appendix=args['--pred-file-appendix'],
        ambifc_subset=args['--subset']
    )
    if args['--eval']:
        pred_directory: str = join(config.get_prediction_directory())
        evaluate_all_veracity_prediction(
            prediction_directory=pred_directory,
            predictions_file=prediction_file,
            split=args['<split>'],
            ambifc_subset=args['--subset'],
            overwrite=args['--overwrite']
        )


if __name__ == "__main__":
    args = docopt(__doc__)
    main(args)

