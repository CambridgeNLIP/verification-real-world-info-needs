import argparse
import csv
import os
import sys
from collections import defaultdict

import boto3
import pymongo
from bson import ObjectId

from annotation.annotation_service import qualify_worker, disqualify_worker
from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data.redirect_db import RedirectDatabase
from annotation.data_service import DataService
from mturk.create_qual import create_qualification_type
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




if __name__ == "__main__":

    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )

    main_qualification = create_qualification_type(mturk,
                                                   "Wikipedia Evidence Finding: FULL ANNOTATION - Qualification Granted")

    freeze_qualification = create_qualification_type(mturk, "Wikipedia Evidence Finding: Soft Block [Awaiting Review]")

    status = defaultdict(list)
    acted = defaultdict(list)
    for idx, worker_claim in enumerate(ds.get_claims_for_annotation_final()):
        claim = ds.get_claim(worker_claim[1])
        annotations = ds.annotations.find({"claim":worker_claim[1],"worker_id":worker_claim[0]},sort=[('_id', pymongo.ASCENDING)])
        for anidx, annotation in enumerate(annotations):
            assignment = ds.assignments.find_one(
                {"_id": ObjectId(annotation["assignment_id"])}
                ,sort=[( '_id', pymongo.DESCENDING)])

            acted[worker_claim[0]].append(True)
            if "keep" in annotation:
                status[worker_claim[0]].append(annotation["keep"])




    for worker in status.keys():
        print("check worker {}".format(worker))
        # check if all HITs annotated
        if len(status[worker]) >60: # == len(acted[worker]):
            success_rate = sum([1 if opt else 0 for opt in status[worker]])/len(status[worker])
            worker_id = ds.create_or_get_worker(worker)["_id"]
            print(worker,success_rate)
            #if success_rate >= 0.9:
            #    qualify_worker(mturk, worker, main_qualification)
            #    disqualify_worker(mturk, worker, freeze_qualification, "Full qualification granted")
            #    mturk.notify_workers(
            #        Subject="Qualification successful for Wikipedia Evidence Finding task",
            #        MessageText="Dear Turker. Thank you for your diligent work for the Wikipedia evidence finding HIT you have been working on. "
            #                    "It is our procedure to review a sample of HITs from new workers to ensure that the submissions meet the guidelines we and to help train you if you need it. "
            #                    "I have reviewed a number of your submissions and am happy with your submissions so I am removing the soft-block so that you can continue working on the task. "
            #                    "Happy turking and thank you for your hard work! ",
            #        WorkerIds=[worker]
            #    )
            #    ds.act_worker(worker_id)
            #    ds.unfreeze_worker(worker_id)

            #    print(worker_id)
            #elif success_rate >= 0.4:
            #    qualify_worker(mturk, worker, suspended_qualification, notify=False)
            #    disqualify_worker(mturk, worker, freeze_qualification, "Suspended following manual review")
            #    mturk.notify_workers(
            #        Subject="Qualification unsuccessful for Wikipedia Evidence Finding task",
            #        MessageText="Dear Turker. We have performed a manual review of your HITs for this task. "
            #                    "Your approval rate was lower than our expected threshold for this task. "
            #                    "For your successful HITs we will be paying you and we invite you to try qualifying for this task again in 5 days.",
            #        WorkerIds = [worker]
            #    )
            #    ds.act_worker(worker_id)
            #    ds.suspend_worker(worker_id)
            #else:
            #    qualify_worker(mturk, worker, suspended_qualification, notify=False)
            #    disqualify_worker(mturk, worker, freeze_qualification, "Suspended following manual review")
#
            #    mturk.notify_workers(
            #        Subject="Qualification unsuccessful for Wikipedia Evidence Finding task",
            #        MessageText="Dear Turker. We have performed a manual review of your HITs for this task. "
            #                    "Your approval rate was lower than our expected threshold for this task. "
            #                    "For your successful HITs we will be paying you, but we will restrict access to this task until we upload a new batch of work.",
            #        WorkerIds = [worker]
            #    )
            #    ds.act_worker(worker_id)
            #    ds.block_worker(worker_id)
        else:
            print(worker, len(status[worker]), " out of ", len(acted[worker]), " annotated")
