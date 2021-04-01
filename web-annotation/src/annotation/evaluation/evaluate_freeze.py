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

print("Get claims from workers requiring annotation")
for idx, worker_claim in enumerate(ds.get_claims_for_annotation()):
    print("Claim")
    claim = ds.get_claim(worker_claim[1])
    print(claim)
    print(worker_claim[0])
    annotations = ds.annotations.find({"claim":worker_claim[1],"worker_id":worker_claim[0], "keep":{"$exists":False}},sort=[('_id', pymongo.ASCENDING)])
    print("Found annotations")
    for anidx, annotation in enumerate(annotations):
        times.append(annotation['annotation_time'])

        if "keep" in annotation:
            continue

        print(annotation)
        print("Assignments")
        assignment = ds.assignments.find_one(
            {"_id": ObjectId(annotation["assignment_id"])}
            ,sort=[( '_id', pymongo.DESCENDING)])

        if os.path.exists("data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"],claim['_id'],anidx)):
            with open("data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"],claim['_id'],anidx)) as f:
                lines = f.readlines()
                if lines[0].strip().lower() != "keep:":
                    continue
        else:
            os.makedirs("data/live/freeze/{}".format(assignment["workerId"]), exist_ok=True)

        if worker_claim[0] not in block:
            to_annotate.append("data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"], claim['_id'],anidx))

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
            "N" if assignment is not None and assignment["workerId"] in block else "",
            "auto" if assignment is not None and assignment["workerId"] in block else "",
            "",
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

        with open("data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"],claim['_id'],anidx),"w+") as f:
            f.write(out_text)


print()
print(" ".join(to_annotate))