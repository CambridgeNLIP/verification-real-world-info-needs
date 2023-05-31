import logging
import json
import os
import random
import sys
import unicodedata
from argparse import ArgumentParser
from collections import defaultdict
from operator import itemgetter

import numpy as np
from drqa.retriever.utils import filter_word, normalize
from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def entity(db,doc):
    return db.get_doc_title(doc)


def section(db,doc):
    return db.get_doc_section(doc)


def get_target_count(row):
    return 1 if (row["filename"],row["read_idx"]) in pre_annotated else 2


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

if __name__ == "__main__":
    generated = []
    parser = ArgumentParser()
    parser.add_argument("--in-files", nargs="+", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    args = parser.parse_args()

    db = FEVERDocumentDatabase(args.db)
    ds = DataService()
    claims = defaultdict(set)
    questions = defaultdict(list)

    with open("pre_annotated.json") as f:
        pre_annotated = set(tuple(a) for a in json.load(f))

    instances = []
    for file in args.in_files:
        with open(file) as infile:
            for idx, row in tqdm(enumerate(infile)):
                row = json.loads(row)
                docs = db.get_ids_from_title(row["title"])
                row["actual_documents"] = docs
                row["filename"] = file
                row["read_idx"] = idx
                instances.append(row)
                claims[row["question"]].add(row["claim"])
                questions[row["question"]].append(row)

    duplicates = [(k,v) for k,v in claims.items() if len(v)>1]

    target_count = 2

    final_instances = []
    for question, claims  in claims.items():
        if len(claims)>1:

            final_instances.extend(questions[question])
        else:
            final_instances.append(questions[question][0])

    instances = final_instances
    random.shuffle(instances)
    instances.sort(key=lambda i: len(i["actual_documents"]))

    sections = 0
    processed = 0

    for row in tqdm(instances):
        if len(row["actual_documents"])>0:
            if len(row["actual_documents"]) > 20:
                print(entity(db,row["actual_documents"][0]), len(row["actual_documents"]))
                continue
            processed += 1
            claim_text = set(filter(lambda tok: not filter_word(tok), set(normalize(row["claim"].lower()).split(" "))))

            for doc in row["actual_documents"]:
                doc_text = [l.split("\t")[1] if len(l.split("\t")) >1 else "" for l in db.get_doc_lines(doc)]
                tokens = (" ".join(doc_text)).split(" ")

                token_set = set([a.lower() for a in tokens])
                if not any(tok in token_set for tok in claim_text):
                    continue

                if len(tokens)<10:
                    continue

                sections += 1

                number_of_lines = len([line for line in doc_text if len(line.split(" "))])
                if number_of_lines <= 12:
                    generated.append({
                        "claim_text": row["claim"],
                        "question": row["question"],
                        "file": row["filename"],
                        "idx": row["read_idx"],
                        "wiki": doc,
                        "entity": entity(db,doc),
                        "section": section(db,doc),
                        "annotation_count": 0,
                        "target_annotation_count": get_target_count(row)
                    })

                else:

                    lines = db.get_doc_lines(doc)

                    paragraphs = []
                    paragraph = []
                    start_line = 0

                    for idx,line in enumerate(lines):
                        if len(line.split("\t")) > 1 and len(line.split("\t")[1].strip()):
                            paragraph.append(line)
                        else:
                            paragraphs.append((start_line,idx,paragraph))
                            paragraph = []
                            start_line = idx

                    paragraphs.append((start_line, len(lines)-1, paragraph))

                    running_total = 0
                    current_start = 0
                    current_end = 0

                    for start_line, end_line, para in paragraphs:
                        to_add = len(para)

                        if len(para) == 0:
                            continue

                        if (running_total > 4 and running_total + to_add > 12) or (running_total+to_add)>20:

                            generated.append({
                                "claim_text": row["claim"],
                                "question": row["question"],
                                "file": row["filename"],
                                "idx": row["read_idx"],
                                "wiki": doc,
                                "entity": entity(db, doc),
                                "section": section(db, doc),
                                "start_line": current_start,
                                "end_line": current_end,
                                "annotation_count": 0,
                                "target_annotation_count": get_target_count(row)
                            })

                            current_start = start_line
                            running_total = 0
                        else:
                            current_end = end_line
                            running_total += to_add

                    if current_end > current_start:
                        generated.append({
                            "claim_text": row["claim"],
                            "question": row["question"],
                            "file": row["filename"],
                            "idx": row["read_idx"],
                            "wiki": doc,
                            "entity": entity(db, doc),
                            "section": section(db, doc),
                            "start_line": current_start,
                            "end_line": current_end,
                            "annotation_count": 0,
                            "target_annotation_count": get_target_count(row)
                        })

    for batch in tqdm(batch(generated,n=1000)):
        ds.claims.insert_many(batch)

