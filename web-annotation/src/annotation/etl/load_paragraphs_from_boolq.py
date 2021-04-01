import datetime
import logging
import json
import os
import unicodedata
from argparse import ArgumentParser

from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data.redirect_db import RedirectDatabase
from annotation.data_service import DataService
from util.wiki import get_wiki_clean_db

logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


import boto3
from boto.mturk.question import ExternalQuestion

endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

# Uncomment this line to use in production
# endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

client = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--create-hit", type=bool, default=False)
    args = parser.parse_args()
    ds = DataService()
    db = FEVERDocumentDatabase(args.db)

    redirects_db_loc = os.getenv("REDIRECTS_DB", "data/redirects.db")
    redirects = RedirectDatabase(redirects_db_loc)

    created =  datetime.datetime.now()
    with open(args.in_file) as infile:
        for idx, row in enumerate(tqdm(infile)):
            row = json.loads(row)

            title = row["title"].replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace(":", "-COLON-")
            try:
                lines = get_wiki_clean_db(db, title, redirects)["text"].split("\n")
            except:
                continue

            paragraph_breaks = [len(line.split("\t"))<=1 or len(line.split("\t")[1].strip())==0 for line in lines]
            paragraphs = []
            paragraph = []
            for line,p_break in zip(lines, paragraph_breaks):

                #print(line, p_break)
                if p_break:
                    if len(paragraph):
                        paragraphs.append(paragraph)
                    paragraph = []
                else:
                    paragraph.append(line)

            for paragraph in paragraphs:

                start_line = int(paragraph[0].split("\t")[0])
                end_line = int(paragraph[-1].split("\t")[0])
                inserted = ds.claims.insert({
                    "claim_text": row["claim"],
                    "page": row["title"].replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace(":",
                                                                                                               "-COLON-"),
                    "annotation_count": 0,
                    "start_line": start_line,
                    "end_line":end_line,
                    "series":"boolq-sample-100claims-sandbox1",
                    "created":created
                })


                if args.create_hit:

                    response = client.create_hit(Title="Natural Claim Validation",
                                                 Reward="0.1",
                                                 AssignmentDurationInSeconds=15 * 60,
                                                 LifetimeInSeconds=7*24 * 60 * 60,
                                                 Description="this is a test of the natural claim validation task",
                                                 Question=ExternalQuestion(external_url="https://natural-claims-annotation.jamesthorne.co.uk/mturk?annotationTarget={}".format(str(inserted)),
                                                                           frame_height=0).get_as_xml())

                    hit_type_id = response['HIT']['HITTypeId']
                    hit_id = response['HIT']['HITId']
                    print("\nCreated HIT: {}".format(hit_id))

                    ds.claims.update({"_id":inserted},
                                            {"$set": {"HIT": hit_id}})



