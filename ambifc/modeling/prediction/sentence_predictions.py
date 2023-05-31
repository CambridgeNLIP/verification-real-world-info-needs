from collections import defaultdict
from typing import Dict, Tuple, List, Iterable


def is_sentence_evidence(sample: Dict, sentence_key: str) -> bool:
    """
    Check if the sentence key belongs to the evidence set of the provided instance. Sentences are considered as part of
    the evidence if at least one annotator assigned a non-neutral label.

    :param sample: Full sample from the dataset.
    :param sentence_key: Sentence key to check.
    """
    annotations: List[str] = list(map(lambda x: x['annotation'], sample['sentence_annotations'][sentence_key]))
    non_neutral_annotations: List[str] = list(filter(lambda x: x != 'neutral', annotations))
    return len(non_neutral_annotations) > 0


def get_non_empty_sentence_keys(sample: Dict) -> List[str]:
    """
    Get all sentence keys from a sample in which actual sentences exist (i.e. excluding empty strings.)

    :param sample: Full sample from the dataset.
    """
    non_empty_sentences: Iterable[str] = sample['sentences'].keys()
    non_empty_sentences = filter(lambda x: len(sample['sentences'][x].strip()) > 0, non_empty_sentences)
    return sorted(list(non_empty_sentences), key=lambda x: int(x))


def get_evidence_sentence_list(sample: Dict) -> List[str]:
    """
    Get all sentence keys from a sample which are part of the evidence.

    :param sample: Full sample from the dataset.
    """
    return sorted(list(filter(lambda x: is_sentence_evidence(sample, x), sample['sentence_annotations'].keys())))


def get_oracle_sentence_prediction_dict(samples: List[Dict]) -> Dict[Tuple[int, str], List[str]]:
    """
    Return a dictionary that maps each sample (identified by claim_id and passage_id) to all sentence keys that are part
    of the evidence set.

    :param samples: All samples from the dataset.
    """
    return {
        (sample['claim_id'], sample['wiki_passage']): sorted(list(
            filter(lambda x: is_sentence_evidence(sample, x), sample['sentence_annotations'].keys())
        ), key=lambda x: int(x)) for sample in samples
    }


def get_fulltext_sentence_prediction_dict(samples: List[Dict]) -> Dict[Tuple[int, str], List[str]]:
    """
    Return a dictionary containing all ordered sentence keys (excluding empty strings). Each sample is represented
    as the claim_id and passage_id as key.

    :param samples: All samples from the dataset.
    """
    return {
        (sample['claim_id'], sample['wiki_passage']): sorted(list(
            filter(lambda x: len(sample['sentences'][x].strip()) > 0, sample['sentences'].keys())
        ), key=lambda x: int(x)) for sample in samples
    }


def get_instance_to_predicted_evidence_dict(predicted_sentences: List[Dict]) -> Dict[Tuple[int, str], List[str]]:
    """
    Return a dictionary that maps each sample (claim_id, passage_id) to a list of all predicted evidence sentences
    (ordered) from all individual sentence predictions.

    :param predicted_sentences: All sentence evidence predictions.
    """

    # Map all individual sentence predictions to the same claim_id, passage_id instance.
    sentence_to_prediction: Dict[Tuple[int, str], Dict[str, str]] = defaultdict(lambda: defaultdict(str))
    for prediction in predicted_sentences:
        sample_id: Tuple[int, str] = (prediction['claim_id'], prediction['passage'])
        sentence_key: str = prediction['sentence_key']

        assert sentence_key not in sentence_to_prediction[sample_id]
        sentence_to_prediction[sample_id][sentence_key] = prediction['predicted']

    # Filter out all sentences that were predicted as evidence; sort them.
    result: Dict[Tuple[int, str], List[str]] = dict()
    for sample_id in sentence_to_prediction.keys():
        prediction_map: Dict[str, str] = sentence_to_prediction[sample_id]
        sentence_keys: List[str] = [sent_key for sent_key in sentence_to_prediction[sample_id].keys()]
        keep_sentence_keys: List[str] = list(filter(lambda x: prediction_map[x] != 'neutral', sentence_keys))
        keep_sentence_keys = sorted(keep_sentence_keys, key=lambda x: int(x))
        result[sample_id] = keep_sentence_keys
    return result
