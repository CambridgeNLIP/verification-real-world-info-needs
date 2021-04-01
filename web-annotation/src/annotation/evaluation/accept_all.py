import os

import boto3
import spacy
from bson import ObjectId
from pymongo import MongoClient
from spacy.matcher import PhraseMatcher
from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService



if __name__ == "__main__":

    client = MongoClient(
                'mongodb://%s:%s@%s' % (os.getenv("MONGO_USER"), os.getenv("MONGO_PASS"), os.getenv("MONGO_HOST")))

    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

    # Uncomment this line to use in production
    # endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )


    old_ds = client["annotate_live"]

    annotations = old_ds.annotations.find({"keep":{"$exists":False}})
    for annotation in annotations:
        if "worker_id" in annotation:

            assignment = old_ds.assigments.find_one({"_id": ObjectId(annotation["assignment_id"])})
            print(annotation)
            print(assignment)
            if assignment is not None:
                try:
                    result = mturk.approve_assignment(AssignmentId=assignment["assignmentId"])
                    print("Approved {}".format(assignment["assignmentId"]))
                except:
                    print("Could not approve {}".format(assignment["assignmentId"]))