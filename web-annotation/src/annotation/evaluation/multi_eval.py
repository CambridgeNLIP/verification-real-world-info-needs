import argparse
import csv
import os

import pymongo
from bson import ObjectId

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

to_annotate = []

def get_label(v):
    return "T\t" if v == 1 else "F\t"

block = []
all_keys = {}


def get_keep(annotation):
    if 'keep' in annotation:
        return "Y" if annotation['keep'] else "N"
    else:
        return ''

def get_status(annotation):
    if 'status' in annotation:
        return annotation['status']
    else:
        return ''

def get_message(annotation):
    if 'message' in annotation:
        return annotation['message']
    else:
        return ''


annotations = ds.annotations.find({"keep":False})
assignments = [(annotation, ds.get_assignment_id(annotation["assignment_id"])) for annotation in annotations]

for annotation, assignment in assignments:

    print(annotation)
    print(assignment)


    if assignment is None:
        print("skip")
        print()
        continue

    claim = ds.get_claim(assignment["claim"])
    out_text = "Keep: {} \n" \
               "Status: {}\n" \
               "Message: {}\nUploaded: {}\n\n\n" \
               "Claim: {}\n" \
               "Page: {}\n" \
               "Date: {}\n" \
               "Assignment: {}\n" \
               "WorkerId: {}\n\n" \
               "Time Taken: {}\n\n" \
               "Label: {}\n" \
               "Extra Info: {}".format(
        get_keep(annotation),
        get_status(annotation),
        get_message(annotation),
        "",
        claim["claim_text"],
        claim["entity"],
        annotation["created"],
        assignment["assignmentId"] if assignment is not None else "",
        assignment["workerId"] if assignment is not None else annotation["username"],
        annotation["annotation_time"],
        annotation["submit_type"],
        "     ".join([annotation["certain"],annotation["relevant"],annotation["unsure"]])
    )

    try:
        lines = get_wiki_clean_db(db, claim["wiki"].replace("_"," ").replace("-LRB-","(").replace("-RRB-",")"), redirects)["text"].split("\n")
    except:
        continue

    tev = []

    for k,v in annotation["evidence"].items():
        if v in [1]:
            line = int(k)
            tev.append(lines[(line+claim["start_line"]) if "start_line" in claim else line])


    if len(tev):
        out_text += "\n\nTRUE Evidence:\n\t"
        out_text += "\n\t".join(["page='{}' line={}".format(claim["entity"],e) for e in tev])

    fev = []

    for k, v in annotation["evidence"].items():
        if v in [-1]:
            line = int(k)
            fev.append(lines[(line+claim["start_line"]) if "start_line" in claim else line])

    if len(fev):
        out_text += "\n\nFALSE Evidence:\n\t"
        out_text += "\n\t".join(["page='{}' line={}".format(claim["entity"],e) for e in fev])


    out_text += "\n\nWikipedia Passage:\n\t"
    out_text += "\n\t".join(lines if "start_line" not in claim else lines[claim["start_line"]:claim["end_line"]])

    print(out_text)
    print()
    print()
