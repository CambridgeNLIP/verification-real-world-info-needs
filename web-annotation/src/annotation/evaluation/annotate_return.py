import argparse
import csv
import glob
import os
import re

import boto3
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

specific_text = {
    "tmi": "You selected some information that seemed related, but was also containing large amounts of additional "
           "sentences that doesn't indicate whether the claim is true or false.",
    "wrong": "You selected sentences that seemed to indicate that the claim was true or false, but the label you "
             "assigned was not suitable (false was true or true was false)",
    "spam": "You selected large amounts of information that are not evidence as to whether the claim is true or false.",
    "error": "A label was assigned but no information was recorded from your web browser",
    "auto": "Following a review of your annotations, a large number of your submissions fail to meet the requirements "
            "of the task resulting in your work being blocked."
}


def get_response(response_status, response_claim, response_message):
    if response_status == "auto":
        return specific_text[response_status]
    base_text = "I have performed a manual review of your annotation of the following claim: `{}`.".format(response_claim)
    return "{} {} {}".format(base_text, specific_text[response_status], response_message)


def get_line(lines, param):
    for line in lines:
        if line.strip().startswith(param.strip()):
            return line.strip().replace(param.strip()).strip()
    return None

if __name__ == "__main__":
    endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )

    for filename in glob.glob("data/pilot/annotate/*-*.txt"):
        if "completed" in filename:
            continue

        with open(filename) as f:
            lines = f.readlines()

        matches = re.match(r"data/pilot/annotate/([a-z0-9]+)-([0-9]+).txt", filename)
        claim_id = matches.group(1)

        if not lines[0].strip().lower().startswith("keep:"):
            continue

        keep = lines[0].replace("Keep:","").strip().lower() == "y"
        reject = lines[0].replace("Keep:", "").strip().lower() == "n"
        act = keep or reject
        if not act:
            continue

        claim = ds.get_claim(ObjectId(claim_id))
        amz_worker_id = get_line(lines, "WorkerId:")
        amz_assignment_id = get_line(lines, "Assignment:")

        annotations = ds.annotations.find({"claim": claim_id, "worker_id": amz_worker_id},
                                          sort=[('_id', pymongo.ASCENDING)])

        main_annotation = None
        main_assignment = None
        for annotation in annotations:
            assignment = ds.assignments.find_one(
                {"_id": ObjectId(annotation["assignment_id"])}
                , sort=[('_id', pymongo.DESCENDING)])

            if assignment["assignmentId"] == amz_assignment_id:
                main_annotation = annotation
                main_assignment = assignment
                break

        annotation = main_annotation
        assignment = main_assignment

        if main_annotation is None:
            print("Could not find annotation for {}".format(claim_id))
            continue

        status = lines[1].replace("Status:","").strip().lower()
        message = lines[2].replace("Message:","").strip()
        uploaded = lines[3].replace("Uploaded:","").strip()

        response = get_response(status,claim["claim_text"], message)


        if not len(uploaded):
            result = None
            success = False
            if not keep and reject and response is not None and assignment is not None:
                print("Reject assignment {} for HIT {}".format(assignment["assignmentId"],assignment["hitId"]))
                try:
                    result = mturk.reject_assignment(AssignmentId=assignment["assignmentId"], RequesterFeedback=response)
                    sucess=True
                except Exception as e:
                    print(e)
                    result = e
            elif keep and assignment is not None:
                print("Accept assignment {} for HIT {}".format(assignment["assignmentId"], assignment["hitId"]))
                try:
                    result = mturk.approve_assignment(AssignmentId=assignment["assignmentId"])
                    success= True
                except Exception as e:
                    print(e)
                    result = e

            if result is not None:
                print(result)
                lines[3] = "Uploaded:" + str(result)
                with open("data/pilot/annotate/{}-{}-completed.txt","w+") as f:
                    f.writelines(lines)
                ds.complete_annotation(annotation["_id"], keep, status, message, str(result) if result is not None else None, success)
                os.unlink(filename)
        else:
            if "keep" not in annotation:
                ds.complete_annotation(annotation["_id"], keep, status, message, uploaded, True)