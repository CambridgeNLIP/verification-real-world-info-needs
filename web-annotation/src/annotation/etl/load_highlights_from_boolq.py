import datetime
import logging
import json
import os
import unicodedata
from argparse import ArgumentParser

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


import boto3
from boto.mturk.question import ExternalQuestion



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--create-hit", action='store_true', default=False)
    parser.add_argument("--ext-url",default='https://natural-claims-annotation.jamesthorne.co.uk/mturk/variant/1')
    args = parser.parse_args()
    ds = DataService()

    if args.create_hit:
        endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

        # Uncomment this line to use in production
        # endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

        client = boto3.client(
            'mturk',
            endpoint_url=endpoint_url, )


    created =  datetime.datetime.now()

    with open(args.in_file) as infile:
        for idx, row in enumerate(infile):
            row = json.loads(row)
            inserted = ds.claims.insert({
                "claim_text": row["claim_text"] if "claim_text" in row else row["claim"],
                "page": row["title"].replace(" ","_").replace("(","-LRB-").replace(")","-RRB-").replace(":","-COLON-"),
                "highlights": row["predicted_evidence"],
                "annotation_count": 0,
                "series": "boolq-sample-100claims-pilot-series3-variant-1",
                "variant": 1,
                "created": created
            })


            if args.create_hit:
                response = client.create_hit(Title="Natural Claim Validation Pilot Study (Variant 3b)",
                                             Reward="0.25",
                                             MaxAssignments=2,
                                             AssignmentDurationInSeconds=1 * 60 * 60,
                                             LifetimeInSeconds=6 * 60 * 60,
                                             Description="Select evidence from the Wikipedia page that can be used to support or refute a simple claim.",
                                             Question=ExternalQuestion(external_url="{}?annotationTarget={}".format(args.ext_url,str(inserted)),
                                                                       frame_height=0).get_as_xml())

                hit_type_id = response['HIT']['HITTypeId']
                hit_id = response['HIT']['HITId']
                print("\nCreated HIT: {}".format(hit_id))

                ds.claims.update({"_id":inserted},
                                        {"$set": {"HIT": hit_id}})




