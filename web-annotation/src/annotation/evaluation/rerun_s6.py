import argparse
import csv
import os

import boto3
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

def get_response(status,claim):
    return status in ["auto","spam","tmi","wrong"]

def get_label(v):
    return "T\t" if v == 1 else "F\t"


endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

mturk = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,)


def get_assignment(lines):
    for idx,line in enumerate(lines):
        if line.startswith("Assignment:"):
            return idx
    return None

for idx, claim in enumerate(ds.claims.find({})):
    annotations = ds.annotations.find({"claim":claim["_id"]})


    claim_status = {}
    for anidx, annotation in enumerate(annotations):
        try:
            print("")
            print("")
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
                         "assignmentId": assignment_id})

                    keep = lines[0].replace("Keep:","").strip().lower() == "y"
                    status = lines[1].replace("Status:","").strip().lower()
                    hit = assignment["hitId"]
                    print(hit)



                    response = get_response(status,claim["claim_text"])

                    if not keep and response and assignment is not None:
                        print("Was rejected")
                        if(claim["HIT"] == hit):
                            print("Not rerun")
                            ds.claims.update({"_id":claim["_id"]},{"$addToSet":{"oldHits":claim["HIT"]},"$unset":{"HIT":""}})


                else:
                    raise Exception(lines)
        except Exception as e:
            print(e)
            pass

