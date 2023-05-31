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
    if status == "tmi":
        return "I have performed a manual review of your annotation of the following claim: `{}`. you selected some information that seemed related, but was also containing large amounts of additional sentences that doesn't indicate whether the claim is true or false ".format(claim)
    if status == "wrong":
        return "I have performed a manual review of your annotation of the following claim: `{}`. you selected sentences that seemed to indacte that the claim was true or false, but the label you assigned was not suitable (false was true or true was false)".format(claim)
    if status == "spam":
        return "I have performed a manual review of your annotation of the following claim: `{}`. you selected large amounts of information that are not evidence as to whether the claim is true or false.".format(claim)
    if status == "error":
        return "I have performed a manual review of your annotation of the following claim: `{}`. A label was assigned but no information was recorded from your web browser".format(claim)
    if status == "auto":
        return "Following a review of your annotations, a large number of your subissions fail to meet the requirements of the task resulting in your work being blocked. "

def get_label(v):
    return "T\t" if v == 1 else "F\t"


endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

mturk = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,)

for idx, claim in enumerate(ds.claims.find({})):
    annotations = ds.annotations.find({"claim":claim["_id"]})
    for anidx, annotation in enumerate(annotations):
        with open("data/pilot/series6/{}-{}.txt".format(claim["_id"],anidx),"r") as f:
            lines = f.readlines()
            if lines[0].strip().startswith("Keep:") and lines[1].strip().startswith("Status:"):

                assignment = ds.assignments.find_one(
                    {"claim": claim["_id"],
                     "sessionId": annotation["username"],
                     "assignmentId": {"$ne": "ASSIGNMENT_ID_NOT_AVAILABLE"}}
                    , sort=[('_id', pymongo.DESCENDING)])


                keep = lines[0].replace("Keep:","").strip().lower() == "y"
                reject = lines[0].replace("Keep:", "").strip().lower() == "n"

                status = lines[1].replace("Status:","").strip().lower()

                response = get_response(status,claim["claim_text"])

                if not keep and reject and response is not None and assignment is not None:
                    print("Reject assignment {} for HIT {}".format(assignment["assignmentId"],assignment["hitId"]))
                    try:
                        print(mturk.reject_assignment(AssignmentId=assignment["assignmentId"], RequesterFeedback=response))
                    except Exception as e:
                        print(e)
                elif keep and assignment is not None:
                    print("Accept assignment {} for HIT {}".format(assignment["assignmentId"], assignment["hitId"]))
                    try:
                        print(mturk.approve_assignment(AssignmentId=assignment["assignmentId"]))
                    except Exception as e:
                        print(e)


            else:
                raise Exception(lines)

