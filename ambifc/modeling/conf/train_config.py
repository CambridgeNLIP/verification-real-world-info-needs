from typing import Dict, Optional, Tuple


class TrainConfig:

    EVIDENCE_SELECTION_FULL = 'full'
    EVIDENCE_SELECTION_ORACLE = 'oracle'

    EVIDENCE_SELECTION_SAMPLE_IF_NO_EVIDENCE_FOR_NEUTRAL = 'no-evidence-sample-for-neutral'
    SAMPLE_ALWAYS_NEUTRAL_EVIDENCE = 'always-sample-for-for-neutral'

    def __init__(self, config: Dict):
        self.config: Dict = config
        self.evidence_sampling_strategy: str = self.config['evidence_sampling_strategy']
        self.batch_size: int = self.config['batch_size']
        self.batch_size_accumulation: int = self.config['batch_size_accumulation']
        self.epochs: int = self.config['epochs']
        self.seed: int = self.config['seed']
        self.lr: int = self.config['lr']
        self.best_metric_name: str = self.config['best_metric_name']
        self.validate()

    def is_limit_number_training_samples(self) -> bool:
        return 'limit_training_samples_size' in self.config and self.config['limit_training_samples_size'] is not None

    def get_max_number_training_samples_params(self) -> Tuple[int, int]:
        return self.config['limit_training_samples_size'], self.seed

    def get_evidence_sampling_strategy(self) -> str:
        return self.evidence_sampling_strategy

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch_size_accumulation(self) -> int:
        return self.batch_size_accumulation

    def get_epochs(self) -> int:
        return self.epochs

    def get_seed(self) -> int:
        return self.seed

    def get_lr(self) -> float:
        return self.lr

    def get_best_metric_name(self) -> str:
        return self.best_metric_name

    def get_min_max_sampling_sentences(self) -> Tuple[Optional[int], Optional[int]]:
        min_sentences: Optional[int] = self.config.get('sampling_min_sentences', None)
        max_sentences: Optional[int] = self.config.get('sampling_max_sentences', None)
        return min_sentences, max_sentences

    def validate(self):
        assert self.get_best_metric_name() is not None
        assert self.get_seed() is not None
        assert self.get_lr() > 0
        assert self.get_batch_size() >= 1
        assert self.get_batch_size_accumulation() >= 1
        assert self.get_epochs() >= 1
        assert self.get_evidence_sampling_strategy() in {
            TrainConfig.EVIDENCE_SELECTION_SAMPLE_IF_NO_EVIDENCE_FOR_NEUTRAL,
            TrainConfig.SAMPLE_ALWAYS_NEUTRAL_EVIDENCE,
            TrainConfig.EVIDENCE_SELECTION_FULL,
            TrainConfig.EVIDENCE_SELECTION_ORACLE,
            None
        }

        if self.get_evidence_sampling_strategy() == TrainConfig.EVIDENCE_SELECTION_SAMPLE_IF_NO_EVIDENCE_FOR_NEUTRAL:
            min_sentences, max_sentences = self.get_min_max_sampling_sentences()
            assert min_sentences is not None and min_sentences >= 0
            assert max_sentences is not None and max_sentences >= 0
