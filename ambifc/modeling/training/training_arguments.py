import os.path
import shutil

from transformers import TrainingArguments

from ambifc.modeling.conf.config import Config


def get_training_args(config: Config, overwrite: bool) -> TrainingArguments:
    model_directory: str = config.model_config.get_model_dir()
    if overwrite and os.path.exists(model_directory):
        shutil.rmtree(model_directory)
    assert not os.path.exists(model_directory)

    return get_training_args_deberta_large_v3(config, overwrite)


def get_training_args_deberta_large_v3(config: Config, overwrite: bool) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.model_config.get_model_dir(),
        overwrite_output_dir=overwrite,
        num_train_epochs=config.training_config.get_epochs(),
        per_device_train_batch_size=config.training_config.get_batch_size(),
        per_device_eval_batch_size=config.training_config.get_batch_size(),
        gradient_accumulation_steps=config.training_config.get_batch_size_accumulation(),
        evaluation_strategy='epoch',
        metric_for_best_model=config.training_config.get_best_metric_name(),
        load_best_model_at_end=True,
        save_strategy='epoch',
        save_total_limit=1,
        seed=config.training_config.get_seed(),
        learning_rate=config.training_config.get_lr(),
        eval_accumulation_steps=1,
        report_to=None
    )
