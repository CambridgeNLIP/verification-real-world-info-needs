import argparse
import csv
import os
from copy import copy

import boto3
import pymongo
from bson import ObjectId

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService

ds = DataService()


def get_new_label(correction):
    first = correction["correction_type"].lower().split()[0]
    if first == "f":
        return "false"
    elif first == "t":
        return "true"
    elif first == "n":
        return "no evidence found"
    else:
        return None


def get_new_evidence(obj, added, removed):
    corrections = obj["correction_type"].lower().split()
    evidence = copy(obj["evidence"])

    for correction in corrections:

        if correction[0] == "+":
            correction = correction.replace("+","")
            label = None
            if "t" in correction:
                label = +1
                correction = correction.replace("t","")

            elif "f" in correction:
                label = -1
                correction = correction.replace("f","")

            added.append(correction)
            if label is not None:
                evidence[correction] = label

        elif correction[0] == "-":
            correction = correction.replace("-","")
            evidence[correction] = 0
            removed.append(correction)

    return evidence

if __name__ == "__main__":
    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )

    db = FEVERDocumentDatabase("newdb/new.db")

    corrections = ds.annotations.find({ "correction_type":{"$ne":None}, "original_submit_type":{"$exists":False}, "original_evidence": {"$exists":False} })

    for correction in corrections:
        if not len(correction["correction_type"]):
            continue

        print(correction["_id"])
        claim = ds.get_claim(correction["claim"])

        new_label = get_new_label(correction)

        added = []
        removed = []
        new_evidence = get_new_evidence(correction,added,removed)

        message = "Hi, thank you for working on this evidence finding task HIT. " \
                  "It's our policy to review a sample of HITs from new annotators. " \
                  "I've been reviewing your annotations and I am very happy with the quality. " \
                  "I've spotted a minor issue with your submission for one of the claims. " \
                  "Rather than reject this, I'll be making a correction on our end -- this is to preserve your accept rate and to help train you for future HITs. " \
                  "For the claim \"{}\", I found the following issue: \"{}\". ".format(claim["claim_text"], correction["message"])

        message += "I've corrected the claim by making the following changes: "
        if added:
            message += "I added these sentences as evidence: \""
            for i in added:
                message += db.get_doc_lines(claim["wiki"])[int(i)].split("\t")[1]
            message += "\". "

        if removed:
            message += "I removed these sentences from the evidence: \""
            for i in removed:
                message += db.get_doc_lines(claim["wiki"])[int(i)].split("\t")[1]
            message += "\". "

        if new_label is not None:

            message += "changing the label to {}. ".format(new_label)


        message += "Hope this helps clarify the task for you. "
        message += "If you have any queries, please feel free to respond to this email. "
        message += "Thank you for your diligence on this task and Happy Turking!"

        print(len(message))
        print(correction["_id"])


        print()

        if new_evidence is not None and "original_evidence" not in correction:
            ds.annotations.update_one({"_id": correction["_id"]},
                                      {"$set": {"evidence": new_evidence,
                                                "original_evidence": correction["evidence"]}})

        if new_label is not None and "original_submit_type" not in correction:
            ds.annotations.update_one({"_id": correction["_id"]}, {
                "$set": {"submit_type": new_label, "original_submit_type": correction["submit_type"]}})

        ds.annotations.update_one({"_id": correction["_id"]}, {"$set": {"keep": True}})

        try:
            status = mturk.notify_workers(Subject="Correction notice for one of your HITs for Wikipedia Evidence Finding Task",
                                MessageText=message,
                                WorkerIds=[correction["worker_id"]]
            )


            print(status)

        except Exception as e:
            print(e)
