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


if __name__ == "__main__":
    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )

    for filename in glob.glob("data/live/anno/*/*-*.txt"):

        worker_id = filename.split("/")[-2]
        claim_id = filename.split("/")[-1].split("-")[0]
        anidx = filename.split("/")[-1].split("-")[1].replace(".txt","")


        assignment = ds.assignments.find_one(
            {"claim": ObjectId(claim_id),
             "workerId": worker_id}
            ,sort=[( '_id', pymongo.DESCENDING)])

        annotation = ds.annotations.find_one({
            "assignment_id": str(assignment["_id"]),
            "worker_id": worker_id,
            "claim": ObjectId(claim_id)
        })

        claim = ds.claims.find_one({"_id":ObjectId(claim_id)})
        
        with open(filename) as f:
            lines = f.readlines()
        if not lines[0].strip().lower().startswith("keep:"):
            continue

        keep = lines[0].replace("Keep:","").strip().lower() == "y"
        reject = lines[0].replace("Keep:", "").strip().lower() == "n"
        error = lines[0].replace("Keep:", "").strip().lower() == "x"
        correction = lines[0].replace("Keep:","").strip().lower() == "c"
        resubmit = lines[0].replace("Keep:", "").strip().lower() == "r"

        act = keep or reject or error or correction or resubmit

        if not act:
            continue

        status = lines[1].replace("Status:","").strip().lower()
        message = lines[2].replace("Message:","").strip()
        uploaded = lines[3].replace("Uploaded:","").strip()

        correction_type = None

        if error:
            status = "error"
        elif resubmit:
            status = "resubmit"
        elif correction:
            correction_type = status
            status = "correction"

        response = None

        if reject:
            try:
                response = get_response(status,claim["claim_text"], message)
            except:
                print("invalid response in data/live/anno/{}/{}-{}.txt".format(assignment["workerId"], claim['_id'],anidx))
                continue


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
            elif (error or keep or resubmit or correction) and assignment is not None:
                print("Accept assignment {} for HIT {}".format(assignment["assignmentId"], assignment["hitId"]))
                try:
                    result = mturk.approve_assignment(AssignmentId=assignment["assignmentId"])
                    success= True
                except Exception as e:
                    print(e)
                    result = e

            if result is not None:
                lines[3] = "Uploaded:" + str(result)

                try:
                    if resubmit:
                        keep = False

                    ds.complete_annotation(annotation["_id"], keep, status, correction_type, message, str(result) if result is not None else None, success)
                    os.unlink("data/live/anno/{}/{}-{}.txt".format(assignment["workerId"],claim['_id'], anidx))
                except:
                    with open("data/live/anno/{}/{}-{}-complete.txt".format(assignment["workerId"],
                                                                              claim['_id'], anidx), "w+") as f:
                        f.writelines(lines)

        else:
            if "keep" not in annotation:
                ds.complete_annotation(annotation["_id"], keep, status, None, message, uploaded, True)