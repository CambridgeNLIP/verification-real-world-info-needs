import json
import math
import logging
import sys
import nltk
from argparse import ArgumentParser
from copy import copy
from multiprocessing.pool import ThreadPool
from typing import List, Set
from drqascripts.retriever.build_tfidf_lines import OnlineTfidfDocRanker
from spellchecker import SpellChecker
from tqdm import tqdm
from annotation.data.fever_db import FEVERDocumentDatabase
from dataset.convert_predictions import normalize_text_to_title

logger = logging.getLogger(__name__)


def get_valid_sentences_from_pages(db:FEVERDocumentDatabase, found: Set):
    sents = {}
    for page in found:
        try:
            lines = db.get_doc_lines(page)
            for line in lines:
                bits = line.split('\t')
                if len(bits) > 1:
                    txt = bits[1].strip()
                    if len(txt):
                        sents[(page, int(bits[0]))] = txt
        except Exception as e:
            print(e)
            pass

    return sents


def get_best_sentences(args, sents_dict, claim):
    if len(sents_dict):
        keys, lines = zip(*list(sents_dict.items()))
        ranker = OnlineTfidfDocRanker(args,lines)
        return [(keys[idx],lines[idx]) for idx, _ in zip(*ranker.closest_docs(claim,args.num_sents))]
    return []

sc = SpellChecker()
def process_instance(args, instance):
    files = []
    if "predicted_documents" in instance:
        found = set(instance["predicted_documents"])
    else:
        found = {normalize_text_to_title(instance["title"])}

    valid = get_valid_sentences_from_pages(db, found)
    claim = instance["claim"]
    toks = nltk.word_tokenize(claim)
    unknown = sc.unknown([w for w in toks])

    best = get_best_sentences(args, valid, " ".join([sc.correction(w) if w in unknown else w for w in toks]))

    for max_tokens in args.num_tokens:
        num_tokens = 0
        predicted_evidence = []
        for key, sent in best:
            num_tokens += len(sent.split())
            predicted_evidence.append(key)
            if num_tokens >= max_tokens:
                break
        ret_inst = copy(instance)
        ret_inst["predicted_evidence"] = predicted_evidence
        files.append(ret_inst)

    return files


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s')
    logger.info("FEVER TF-IDF Sentence Selector")

    parser = ArgumentParser()
    parser.add_argument("--db")
    parser.add_argument("--in-file")
    parser.add_argument("--out-file")
    parser.add_argument("--spellcheck", action='store_true')
    parser.add_argument('--ngram', type=int, default=2,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--tokenizer', type=str, default='simple',
                        help=("String option specifying tokenizer type to use "
                              "(e.g. 'corenlp')"))
    parser.add_argument("--num-sents",type=int, default=1000)
    parser.add_argument("--num-tokens",type=int, default=[1000], nargs="+")
    args = parser.parse_args()

    logger.info("Processing file {}".format(args.in_file))
    logger.info("Load document database {}".format(args.db))
    db = FEVERDocumentDatabase(args.db)

    out_files = [open(args.out_file.replace("@",str(a)),"w+") for a in args.num_tokens ]

    instances = []
    with open(args.in_file) as f:
        for line in f:
            instance = json.loads(line)
            instances.append(instance)

    logger.info("Running IR")
    with ThreadPool() as p:
        for line in tqdm(p.imap(lambda a: process_instance(args,a), instances), total=len(instances)):
            for out_file, data in zip(out_files, line):
                out_file.write(json.dumps(data) + "\n")
                out_file.flush()
