import os

import spacy
from pymongo import MongoClient
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService


def find_matches(old_wiki_lines, new_wiki_lines):
    patterns = [nlp.make_doc(line.split("\t")[1]) for line in old_wiki_lines]
    matcher.add("TerminologyList", None, *patterns)

    docs = (nlp(line.split("\t")[1]) for line in new_wiki_lines if len(line.split("\t"))>1)
    matches = (matcher(doc) for doc in docs)

    matched = set()
    for idx,m in enumerate(matches):
        for match_id, start, end in m:
            matched.add(idx)
    return matched


if __name__ == "__main__":

    nlp = spacy.load('en_core_web_sm')
    matcher = PhraseMatcher(nlp.vocab)


    client = MongoClient(
                'mongodb://%s:%s@%s' % (os.getenv("MONGO_USER"), os.getenv("MONGO_PASS"), os.getenv("MONGO_HOST")))

    old_ds = client["annotate_live"]
    new_ds = client["simple_live"]


    db = FEVERDocumentDatabase("data/fever-sections.db")
    db_new = FEVERDocumentDatabase("newdb/new.db")

    annotations = old_ds.annotations.find({"keep":{"$ne":False}})
    claims = set()

    for annotation in annotations:
        claims.add(annotation["claim"])


    claims = old_ds.claims.find({"_id": {"$in": list(claims)} })
    claims_dict = {}
    for claim in claims:
        claims_dict[claim["_id"]] = claim

    complete = 0
    annotations = old_ds.annotations.find({"keep": {"$ne": False}})
    total =0
    for annotation in tqdm(annotations,total=annotations.count()):
        if annotation["submit_type"] in ["true","false"]:
            total+=1
            claim = claims_dict[annotation["claim"]]
            lines = db.get_full_doc_lines_and_tile(claim["wiki"])

            file, idx = claim["file"], claim["idx"]
            match_claims = new_ds.claims.find({"file":file,"idx":idx})

            for match in match_claims:
                new_wiki_lines = db_new.get_doc_lines(match["wiki"])

                pos_lines = [(int(k),v) for k,v in annotation["evidence"].items() if v != 0]
                line_text = lines[2]

                matches = find_matches([line_text[pos_line] for pos_line,_ in pos_lines], new_wiki_lines)

                if " ".join(line_text[pos_line].split("\t")[1] for pos_line,_ in pos_lines if len(line_text[pos_line].split("\t"))>1) \
                        == " ".join(new_wiki_lines[match].split("\t")[1] for match in matches if len(new_wiki_lines[match].split("\t"))>1) and not \
                        (-1 not in set(annotation["evidence"].values()) and 1 not in set(annotation["evidence"].values())):

                    print("complete")
                    complete +=1



            print(complete,total)