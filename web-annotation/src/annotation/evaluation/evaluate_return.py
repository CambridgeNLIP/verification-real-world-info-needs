import argparse
import csv
import os

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

    for idx, worker_claim in enumerate(ds.get_claims_for_annotation()):
        claim = ds.get_claim(worker_claim[1])
        annotations = ds.annotations.find({"claim":worker_claim[1],"worker_id":worker_claim[0],"keep":{"$exists":False}},sort=[('_id', pymongo.ASCENDING)])

        print("Annotator {} has {} to check".format(worker_claim[0], annotations.count()))
        for anidx, annotation in enumerate(annotations):
            assignment = ds.assignments.find_one(
                {"_id": ObjectId(annotation["assignment_id"])}
                ,sort=[( '_id', pymongo.DESCENDING)])

            if os.path.exists("data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"], claim['_id'],anidx)):
                with open("data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"],claim['_id'],anidx)) as f:
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
                        print("invalid response in data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"], claim['_id'],anidx))
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
                            os.unlink("data/live/freeze/{}/{}-{}.txt".format(assignment["workerId"],claim['_id'], anidx))
                        except:
                            with open("data/live/freeze/{}/{}-{}-complete.txt".format(assignment["workerId"],
                                                                                      claim['_id'], anidx), "w+") as f:
                                f.writelines(lines)

                else:
                    if "keep" not in annotation:
                        ds.complete_annotation(annotation["_id"], keep, status, None, message, uploaded, True)