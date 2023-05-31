import random
from typing import Iterable, Dict, List, Optional

from ambifc.modeling.conf.train_data_config import TrainDataConfig
from ambifc.util.fileutil import read_jsonl_from_dir

SPLITS: List[str] = ['train', 'dev', 'test']


def get_samples_for_ambifc_subset(
        ambifc_subset: str, split: str, data_directory: str,
        max_num: Optional[int] = None, max_num_shuffle_seed: Optional[int] = None
) -> List[Dict]:
    """
    Get all samples from AmbiFC given the defined subset and split.

    :param ambifc_subset: Define the subset.
    :param split: train/dev/test.
    :param data_directory: Directory containing the data files.
    :param max_num: Optional number to limit number of samples.
    :param max_num_shuffle_seed: Optional seed if samples must be samples based on <max_num>.
    """
    assert ambifc_subset in {
        TrainDataConfig.SUBSET_CERTAIN_ONLY,
        TrainDataConfig.SUBSET_ALL_AMBIFC,
        TrainDataConfig.SUBSET_UNCERTAIN_ONLY,
        TrainDataConfig.SUBSET_UNCERTAIN_ONLY_ALL,
        TrainDataConfig.SUBSET_CERTAIN_FIVE_PLUS
    }, f'Unknown subset: "{ambifc_subset}"'
    assert split in SPLITS
    if ambifc_subset == TrainDataConfig.SUBSET_CERTAIN_ONLY:
        samples = _get_samples_for_certain(split, data_directory)
    elif ambifc_subset == TrainDataConfig.SUBSET_ALL_AMBIFC:
        samples = _get_samples_for_ambifc(split, data_directory)
    elif ambifc_subset == TrainDataConfig.SUBSET_UNCERTAIN_ONLY:
        samples = _get_samples_for_uncertain_five_plus(split, data_directory)
    elif ambifc_subset == TrainDataConfig.SUBSET_UNCERTAIN_ONLY_ALL:
        samples = _get_samples_for_uncertain_all(split, data_directory)
    elif ambifc_subset == TrainDataConfig.SUBSET_ALL_ANNOTATED:
        samples = _get_samples_for_all(split, data_directory)
    elif ambifc_subset == TrainDataConfig.SUBSET_CERTAIN_FIVE_PLUS:
        samples = _get_samples_for_certain_five_plus(split, data_directory)
    else:
        raise ValueError(ambifc_subset)

    # If a maximum number of samples is specified, only select a subset.
    if max_num is not None:
        assert max_num_shuffle_seed is not None
        random.seed(max_num_shuffle_seed)
        random.shuffle(samples)
        samples = samples[:max_num]
    return samples


def _get_samples_for_uncertain_five_plus(split: str, data_directory: str) -> List[Dict]:
    """
    Get all samples from AmbiFC (uncertain). The used subset only includes samples with 5+ annotations.
    """
    uncertain_samples: Iterable[Dict] = read_jsonl_from_dir(data_directory, f'{split}.uncertain.jsonl')
    uncertain_samples = filter(lambda sample: len(sample['passage_annotations']) >= 5, uncertain_samples)
    return list(uncertain_samples)


def _get_samples_for_uncertain_all(split: str, data_directory: str) -> List[Dict]:
    """
    Get all samples with uncertain/relevant annotations. This is not restricted to 5+ annotations.
    """
    uncertain_samples: Iterable[Dict] = read_jsonl_from_dir(data_directory, f'{split}.uncertain.jsonl')
    return list(uncertain_samples)


def _get_samples_for_certain(split: str, data_directory: str) -> List[Dict]:
    """
    Get all samples from AmbiFC (certain).
    """
    return list(read_jsonl_from_dir(data_directory, f'{split}.certain.jsonl'))


def _get_samples_for_certain_five_plus(split: str, data_directory: str) -> List[Dict]:
    """
    Get all samples from AmbiFC (certain) with 5+ annotations.
    """
    samples: List[Dict] = _get_samples_for_certain(split, data_directory)
    return list(filter(lambda sample: len(sample['passage_annotations']) >= 5, samples))


def _get_samples_for_ambifc(split: str, data_directory: str) -> List[Dict]:
    """
    Get all samples for experiments from AmbiFC. This includes AmbiFC (certain) and AmbiFC (uncertain).
    """
    # All samples from the certain subset.
    certain_samples: List[Dict] = _get_samples_for_certain(split, data_directory)

    # All samples from the uncertain subset with 5+ annotations
    uncertain_samples: Iterable[Dict] = read_jsonl_from_dir(data_directory, f'{split}.uncertain.jsonl')
    uncertain_samples = filter(lambda sample: len(sample['passage_annotations']) >= 5, uncertain_samples)
    return certain_samples + list(uncertain_samples)


def _get_samples_for_all(split: str, data_directory: str) -> List[Dict]:
    """
    Get all samples without any filtering based on subset or number of annotations.
    """
    # All samples from the certain subset.
    certain_samples: List[Dict] = _get_samples_for_certain(split, data_directory)

    # All samples from the uncertain subset with 5+ annotations
    uncertain_samples: Iterable[Dict] = read_jsonl_from_dir(data_directory, f'{split}.uncertain.jsonl')
    return certain_samples + list(uncertain_samples)
