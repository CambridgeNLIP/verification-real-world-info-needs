import logging
import json
import os
import unicodedata
from argparse import ArgumentParser

from drqa.retriever.utils import filter_word, normalize

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def entity(db,doc):
    return db.get_doc_title(doc)

def section(db,doc):
    return db.get_doc_section(doc)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--full_db", type=str, required=True)
    args = parser.parse_args()
    ds = DataService()
    db = FEVERDocumentDatabase(args.db)
    fdb = FEVERDocumentDatabase(args.full_db)
    with open(args.in_file) as infile:
        for idx, row in enumerate(infile):
            if idx%2 == 1:
                continue

            row = json.loads(row)

            for doc in row["predicted_documents"]:
                doc_title = entity(fdb,doc)
                claim_text = set(
                    filter(lambda tok: not filter_word(tok), set(normalize(row["claim"].lower()).split(" "))))

                for doc in db.get_ids_from_title(doc_title):
                    doc_text = [l.split("\t")[1] if len(l.split("\t")) > 1 else "" for l in db.get_doc_lines(doc)]
                    tokens = set(normalize(" ".join(doc_text)).lower().split(" "))

                    if not any(tok in tokens for tok in claim_text):
                        continue

                    tokens = (" ".join(doc_text)).split(" ")
                    if len(tokens) < 10:
                        continue

                    number_of_lines = len([line for line in doc_text if len(line.split(" "))])

                    if number_of_lines <= 12:
                        ds.claims.insert({
                            "claim_text": row["claim"],
                            "wiki": doc,
                            "entity": entity(db, doc),
                            "section": section(db, doc),
                            "annotation_count": 0
                        })

                    else:

                        lines = db.get_doc_lines(doc)

                        paragraphs = []
                        paragraph = []
                        start_line = 0

                        for idx, line in enumerate(lines):
                            if len(line.split("\t")) > 4 and len(line.split("\t")[1].strip()):
                                paragraph.append(line)
                            else:
                                paragraphs.append((start_line, idx, paragraph))
                                paragraph = []
                                start_line = idx

                        paragraphs.append((start_line, len(lines) - 1, paragraph))

                        running_total = 0
                        current_start = 0
                        current_end = 0

                        for start_line, end_line, para in paragraphs:
                            to_add = len(para)

                            if len(para) == 0:
                                continue

                            if (running_total > 4 and running_total + to_add > 12) or (running_total+to_add)>20:

                                ds.claims.insert({
                                    "claim_text": row["claim"],
                                    "wiki": doc,
                                    "entity": entity(db, doc),
                                    "section": section(db, doc),
                                    "start_line": current_start,
                                    "end_line": current_end,
                                    "annotation_count": 0
                                })

                                current_start = start_line
                                running_total = 0
                            else:
                                current_end = end_line
                                running_total += to_add

                        if current_end > current_start:
                            ds.claims.insert({
                                "claim_text": row["claim"],
                                "wiki": doc,
                                "entity": entity(db, doc),
                                "section": section(db, doc),
                                "start_line": current_start,
                                "end_line": current_end,
                                "annotation_count": 0
                            })