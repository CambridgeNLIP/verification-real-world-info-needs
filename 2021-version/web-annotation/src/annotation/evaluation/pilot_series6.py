import argparse
import csv
import os

import pymongo

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

rows = []
block = [""]
all_keys = {}
for idx, claim in enumerate(ds.claims.find({})):
    print("Claim")
    annotations = ds.annotations.find({"claim":claim["_id"]},sort=[('_id', pymongo.ASCENDING)])
    print("Found annotations")
    for anidx, annotation in enumerate(annotations):
        times.append(annotation['annotation_time'])

        print(annotation)
        print("Assignments")
        assignment = ds.assignments.find_one(
            {"claim":claim["_id"],
             "sessionId": annotation["username"],
             "assignmentId":{"$ne":"ASSIGNMENT_ID_NOT_AVAILABLE"}}
            ,sort=[( '_id', pymongo.DESCENDING)])


        if assignment is not None:
            ret_obj = {}
            ret_obj["claim_id"] = claim["_id"]
            ret_obj["annotation_id"] = annotation["_id"]
            ret_obj["index"] = anidx
            ret_obj.update(claim)
            ret_obj.update(annotation)
            ret_obj.update(assignment)
            ret_obj.update({
                "keep":"",
                "comment":""
            })
            rows.append(ret_obj)
            all_keys.update({k:1 for k in ret_obj.keys()})


        if os.path.exists("data/pilot/series6/{}-{}.txt".format(claim['_id'],anidx)):
            with open("data/pilot/series6/{}-{}.txt".format(claim['_id'],anidx)) as f:
                lines = f.readlines()
                if lines[0].strip().lower() != "keep:":
                    continue


        if assignment is not None and assignment["workerId"] not in block:
            to_annotate.append("data/pilot/series6/{}-{}.txt".format(claim['_id'],anidx))

        out_text = "Keep: {} \n" \
                   "Status: {}\n\n" \
                   "Claim: {}\n" \
                   "Page: {}\n" \
                   "Date: {}\n" \
                   "Assignment: {}\n" \
                   "WorkerId: {}\n\n" \
                   "Time Taken: {}\n\n" \
                   "Label: {}".format(
            "N" if assignment is not None and assignment["workerId"] in block else "",
            "auto" if assignment is not None and assignment["workerId"] in block else "",
            claim["claim_text"],
            claim["entity"],
            annotation["created"],
            assignment["assignmentId"] if assignment is not None else "",
            assignment["workerId"] if assignment is not None else annotation["username"],
            annotation["annotation_time"],
            annotation["submit_type"]
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

        if not len(tev)+len(fev):
            out_text += "\n\nWikipedia Passage:\n\t"
            out_text += "\n\t".join(lines if "start_line" not in claim else lines[claim["start_line"]:claim["end_line"]])

        with open("data/pilot/series6/{}-{}.txt".format(claim['_id'],anidx),"w+") as f:
            f.write(out_text)

with open("data/pilot/series6/timings.txt","w+") as f:
    f.write('\n'.join(str(t) for t in times))

with open("data/pilot/series6/results.csv", "w+") as f:
    field_names = list(all_keys)
    writer=csv.DictWriter(f,fieldnames=field_names)
    writer.writeheader()
    for r in rows:
        writer.writerow(r)

print()
print(" ".join(to_annotate))