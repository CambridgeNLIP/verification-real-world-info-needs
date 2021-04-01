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

#client = boto3.client(
#    'mturk',
#    endpoint_url=endpoint_url,)


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

            paragraph_breaks = [len(line.split("\t"))<=2 and len(line.split("\t")[1].strip())>2 and "." not in line for line in lines]

            sections = []
            section = []
            for line,p_break in zip(lines, paragraph_breaks):

                #print(line, p_break)
                if p_break:
                    sections.append(section)
                    section = [line]
                else:

                    section.append(line)



            spares = []
            for section in sections:
                if not len(section):
                    continue
                blank_lines = [len(line.split("\t")) <= 1 or len(line.split("\t")[1].strip()) == 0 for line in
                                    section]

                if sum([1 if line else 0 for line in blank_lines])>3 and "see also" not in section[0].lower():

                    start_line = int(section[0].split("\t")[0])
                    end_line = int(section[-1].split("\t")[0])

                    inserted = ds.claims.insert({
                        "claim_text": row["claim"],
                        "page": row["title"].replace(" ", "_").replace("(", "-LRB-").replace(")", "-RRB-").replace(":",
                                                                                                                   "-COLON-"),
                        "annotation_count": 0,
                        "start_line": start_line,
                        "end_line": end_line,
                        "series": "boolq-sample-100claims-sandbox1",
                        "created": created
                    })
                elif sum([1 if line else 0 for line in blank_lines])<=3:
                    spares.extend(section)


            #print(spares)


