import argparse
import csv
import json
import os

import boto3
import pymongo
from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data.redirect_db import RedirectDatabase
from annotation.data_service import DataService
from util.wiki import get_wiki_clean_db

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, required=True)
args = parser.parse_args()
ds = DataService()
db = FEVERDocumentDatabase(args.db)

redirects_db_loc = os.getenv("REDIRECTS_DB", "data/redirects.db")
redirects = RedirectDatabase(redirects_db_loc)

times = []

def get_label(v):
    return "T\t" if v == 1 else "F\t"

def get_assignment(lines):
    for idx,line in enumerate(lines):
        if line.startswith("Assignment:"):
            return idx
    return None


def get_worker(lines):
    for idx,line in enumerate(lines):
        if line.startswith("WorkerId:"):
            return idx
    return None


claim_annotations = {}

for idx, claim in enumerate(tqdm(ds.claims.find({}))):
    annotations = ds.annotations.find({"claim":claim["_id"]})
    claim_text = claim["claim_text"]
    entity = claim["entity"]
    section = claim["section"]
    wiki = claim["wiki"]

    if claim["claim_text"] not in claim_annotations:
        claim_hits = claim["active_hits"]

        if "oldHits" in claim:
            claim_hits.extend(hit for hit in claim["oldHits"])

        claim_annotations[claim_text] = \
            {"claim":
                 claim_text,
             "metadata":
                 {"hits":claim_hits,
                  "masters": [],
                  "annotated_sections":set()},
             "annotations":
                 [],
             "judgements":
                 []
            }

    claim_annotations[claim_text]["metadata"]["masters"].append(str(claim["_id"]))
    for anidx, annotation in enumerate(annotations):
        with open("data/pilot/series6/{}-{}.txt".format(claim["_id"],anidx),"r") as f:
            lines = f.readlines()
            if lines[0].startswith("Keep:") and lines[1].startswith("Status:"):
                assignment = get_assignment(lines)
                if assignment is None:
                   continue
                assignment_id = lines[assignment].replace("Assignment:","").strip()
                assignment = ds.assignments.find_one(
                    {"claim": claim["_id"],
                     "sessionId": annotation["username"],
                     "assignmentId": assignment_id,

                     }
                )

                if assignment is None:
                    continue

                reject = lines[0].replace("Keep:","").strip().lower() == "n"
                status = lines[1].replace("Status:","").strip().lower()
                hit = assignment["hitId"]


                if reject or len(status):
                    continue


                try:
                    lines = \
                    get_wiki_clean_db(db, claim["wiki"].replace("_", " ").replace("-LRB-", "(").replace("-RRB-", ")"),
                                      redirects)["text"].split("\n")
                except:
                    continue


                claim_annotations[claim_text]["metadata"]["annotated_sections"].add((entity, section))
                claim_annotations[claim_text]["judgements"].append(annotation["submit_type"])
                for k, v in annotation["evidence"].items():
                    if v != 0 and annotation["submit_type"] != "can't tell":
                        line = int(k)
                        sentence_number,sentence = lines[(line + claim["start_line"]) if "start_line" in claim else line].split("\t")
                        claim_annotations[claim_text]["annotations"].append({
                            "page":entity,
                            "section":section if len(section.strip()) else "Introduction",
                            "filename": wiki,
                            "sentence": sentence,
                            "veracity": "LINE_TRUE" if v > 0 else "LINE_FALSE",
                            "sentence_number": sentence_number,
                            "certainty": "HIGH" if annotation["submit_type"].startswith("certainly") else "LOW",
                            "top_level_label": annotation["submit_type"]
                        })


with open("dataset.jsonl","w+") as f:
    for line in claim_annotations.values():
        tmp = line["metadata"]["annotated_sections"]
        line["metadata"]["annotated_sections"] = [(page,section) for page,section in tmp]
        f.write(json.dumps(line)+"\n")


for value in claim_annotations.values():
    print(value)
    print()
