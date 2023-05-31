import argparse
import csv
import glob
import os
import sys

import boto3
import pymongo
from bson import ObjectId

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService

ds = DataService()

times = []
to_annotate = []

def get_label(v):
    return "T\t" if v == 1 else "F\t"

block = []
all_keys = {}

specific_text = {
    "tmi": "You selected some information that seemed related, but was also containing large amounts of additional "
           "sentences that doesn't indicate whether the claim is true or false.",
    "nei": "The evidence you selected was related, but wouldn't be enough to act as evidence to support or refute the claim",
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



if __name__ == "__main__":
    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )

    perfect_matches = 0
    evidence_matches = 0
    label_matches = 0
    for filename in glob.glob("data/live/freeze/*/*-*.txt"):

        worker_id = filename.split("/")[-2]
        claim_id = filename.split("/")[-1].split("-")[0]
        anidx = filename.split("/")[-1].split("-")[1].replace(".txt","")


        assignments = ds.assignments.find(
            {"claim": ObjectId(claim_id)}
            ,sort=[( '_id', pymongo.DESCENDING)])

        assignment_ids = [str(a["_id"]) for a in assignments]


        main_annotation = ds.annotations.find_one({
            "assignment_id": {"$in": assignment_ids},
            "worker_id": worker_id,
            "claim": ObjectId(claim_id),

        })

        other_annotations = ds.annotations.find({
            "assignment_id": {"$in": assignment_ids},
            "worker_id": {"$ne": worker_id},
            "claim": ObjectId(claim_id)
        })


        for annotation in other_annotations:
            if annotation is not None:
                label_ok = annotation["submit_type"] == main_annotation["submit_type"]
                evidence_ok = annotation["evidence"] == main_annotation["evidence"]

                if label_ok:
                    label_matches+=1

                if evidence_ok:
                    evidence_matches +=1

                if label_ok and evidence_ok:
                    perfect_matches +=1

                    ret_assignment = ds.assignments.find_one({"_id": ObjectId(main_annotation["assignment_id"])})
                    print("Accept assignment {} for HIT {}".format(ret_assignment["assignmentId"], ret_assignment["hitId"]))

                    try:
                        result = mturk.approve_assignment(AssignmentId=ret_assignment["assignmentId"])
                        success = True
                    except Exception as e:
                        print(e)
                        result = e

                    ds.complete_annotation(str(main_annotation["_id"]), True, "", "", "",
                                       str(result) if result is not None else None, True)
                    os.unlink(filename)
                    break

        print(label_matches,evidence_matches, perfect_matches)

