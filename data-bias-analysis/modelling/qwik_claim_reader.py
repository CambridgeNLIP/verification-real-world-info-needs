import json
from dataclasses import Field
from typing import Iterable, Optional, Dict
from allennlp.data import DatasetReader, Instance, Tokenizer, TokenIndexer
from allennlp.data.fields import TextField, LabelField

import itertools
import operator

def most_common(L):
  # get an iterable of (item, iterable) pairs
  SL = sorted((x, i) for i, x in enumerate(L))
  # print 'SL:', SL
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  # auxiliary function to get "quality" for an item
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    # print 'item %r, count %r, minind %r' % (item, count, min_index)
    return count, -min_index
  # pick the highest-count/earliest item
  return max(groups, key=_auxfun)[0]

@DatasetReader.register("qwic_claim_only")
class SentenceReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 majority:bool=False,
                 read_classes:Optional[Iterable[str]] = None,
                 lazy: bool = False, cache_directory: Optional[str] = None, max_instances: Optional[int] = None,
                 manual_distributed_sharding: bool = False, manual_multi_process_sharding: bool = False,
                 serialization_dir: Optional[str] = None) -> None:
        super().__init__(lazy, cache_directory, max_instances, manual_distributed_sharding,
                         manual_multi_process_sharding, serialization_dir)

        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.read_classes = read_classes
        self.majority = majority

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path) as f:

            if self.majority and "dev" not in file_path and "test" not in file_path:
                for line in f:
                    nc_claim = json.loads(line)
                    yield from self.get_claim_sentence_common(nc_claim)
            else:

                for line in f:
                    nc_claim = json.loads(line)
                    yield from self.get_claim_sentence(nc_claim)

    def get_claim_sentence(self, claim):
        claim_text = claim['claim']

        for section_id, lines in claim['text'].items():
            if "labels" in claim and self.read_classes and claim['labels'][section_id]['section_label'] not in self.read_classes:
                continue
            yield self.text_to_instance(claim=claim_text,
                                        evidence=None,
                                        label=claim['labels'][section_id]['section_label'] if 'labels' in claim else None)

    def get_claim_sentence_common(self, claim):
        claim_text = claim['claim']
        labels = [claim['labels'][section_id]['section_label'] for section_id, lines in claim['text'].items()]
        mc = most_common(labels)

        yield self.text_to_instance(claim=claim_text,
                                    evidence=None,
                                    label=mc)

    def text_to_instance(self, claim, evidence, label=None) -> Instance:
        fields: Dict[str, Field] = {}

        claim = self._tokenizer.tokenize(claim)
        evidence = self._tokenizer.tokenize(evidence)[1:] if evidence else []
        tokens = claim + evidence

        fields['tokens'] = TextField(tokens, self._token_indexers)
        if label:
            fields['label'] = LabelField(label)

        return Instance(fields)

