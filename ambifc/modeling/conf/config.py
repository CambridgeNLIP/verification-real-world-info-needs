from os.path import join
from typing import Dict, List

from ambifc.modeling.conf.model_config import ModelConfig
from ambifc.modeling.conf.train_config import TrainConfig
from ambifc.modeling.conf.train_data_config import TrainDataConfig


class Config:

    """
    Stores and validates the configuration files.
    """

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.model_config: ModelConfig = ModelConfig(config['model'])
        self.training_config: TrainConfig = TrainConfig(config['training'])
        self.train_data_config: TrainDataConfig = TrainDataConfig(config['data'])

    def get_prediction_directory(self) -> str:
        return join(self.config['predictions']['directory'], self.model_config.get_model_dest())
