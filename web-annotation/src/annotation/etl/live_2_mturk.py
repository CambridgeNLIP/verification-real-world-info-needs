import json
from collections import defaultdict
from datetime import datetime, timedelta
import logging
import os
from argparse import ArgumentParser
from typing import List
import sys

from tqdm import tqdm

"""
    for hit in claim["hits"]:
    expected = hit["expected_num"]
    total_annotations += len(hit_annotations[hit["id"]])
    total_expected += expected

    completed = 0
    rejected = len(rejected_annotations(hit_annotations[hit["id"]]))

    if hit["expires"] <= datetime.now():
        completed += len(non_rejected_annotations(hit_annotations[hit["id"]]))
        hit_shortfall = expected - completed + rejected
    else:
        hit_shortfall = rejected
    total_shortfall += hit_shortfall
"""
from bson import ObjectId

from annotation.data_service import DataService
from mturk.create_qual import create_qualification_type


logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


import boto3
from boto.mturk.question import ExternalQuestion

endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

# Uncomment this line to use in production
#endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

client = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,)


def non_rejected_annotations(annotations) -> List:
    ret = []
    for annotation in annotations:
        if "keep" not in annotation or annotation["keep"]:
            ret.append(annotation)
    return ret


def rejected_annotations(annotations) -> List:
    ret = []
    for annotation in annotations:
        if "keep" in annotation and not annotation["keep"]:
            ret.append(annotation)

    return ret



def get_claim_assignment_shortfall(claim) -> int:
    if "hits" not in claim or len(claim["hits"]) == 0:
        return claim["target_annotation_count"]

    hits = {}
    hit_annotations = defaultdict(list)

    for hit in claim["hits"]:
        hits[hit["id"]] = hit
        hit_annotations[hit["id"]].extend(ds.get_annotations_from_hit(claim, hit["id"]))

    target = claim["target_annotation_count"]
    total_shortfall = 0

    # Go through all HITs
    #   count number of good annotations if expired
    #   count number of annotations awaited if active
    # return target - count

    total_expected = 0
    for hit in claim["hits"]:

        if hit["expires"] <= datetime.now():
            # If hit has expired then return the number of non-rejected annotations
            total_expected += len(non_rejected_annotations(hit_annotations[hit["id"]]))
        else:
            #If hit is still active all claims could be answered (barring rejected ones)
            expected = hit["expected_num"]
            rejected = len(rejected_annotations(hit_annotations[hit['id']]))
            total_expected += expected - rejected


    # Also exclude annotations made on an instance that do not have HITs
    annotations_without_hits = ds.get_annotations_exclude_hits(claim, hit_annotations.keys())
    total_expected += len(non_rejected_annotations(annotations_without_hits))

    shortfall = target - total_expected

    #assert to_make >= 0
    return shortfall


if __name__ == "__main__":


    dry_run = False

    timeout_qualification = create_qualification_type(client, "Wikipedia Evidence Finding: Soft Block [Timeout]")
    pending_qualification = create_qualification_type(client, "Wikipedia Evidence Finding: Soft Block [Awaiting Review]")
    main_qualification = create_qualification_type(client, "Wikipedia Evidence Finding: Quiz 2")
    print(main_qualification)

    test = " ".join(open("qualify.xml").readlines()).strip()
    answers = " ".join(open("qualify_answers.xml").readlines())

    client.update_qualification_type(QualificationTypeId=main_qualification,
                                     Test=test,
                                     AnswerKey=answers,
                                     TestDurationInSeconds=60*30,
                                     RetryDelayInSeconds=60*60*24*4)
    xp = client.get_paginator('list_qualification_requests')
    for qual_req in xp.paginate(
        QualificationTypeId=main_qualification,

    ):
        print(qual_req)


    parser= ArgumentParser()
    parser.add_argument("--ext-url",default="https://annotation-live.jamesthorne.co.uk/mturk/variant/11")
    args = parser.parse_args()

    created =  datetime.now()
    ds = DataService()
    added = 0

    #claim = ds.get_claim(ObjectId("5e8b6702861d99402d0ee284"))
    #print(ds.get_annotations_from_claim(claim))
    #print(get_claim_assignment_shortfall(claim))
    offs = 21886+21329+20763+26040
    cursor = ds.claims.find(no_cursor_timeout=True).skip(offs) #.find({"hits":{"$exists":False}})
    print("Found {}".format(cursor.count()))
    for claim in tqdm(cursor, total=cursor.count()-offs):
        print("Got claim {}".format(claim["_id"]))

        #if claim["annotation_count"] >= claim["target_annotation_count"]:
        #    continue


        assignments_to_make = get_claim_assignment_shortfall(claim)
        print("Got shortfall {}".format(assignments_to_make))
        print()

        if "start_line" in claim and "end_line" in claim:
            if claim["start_line"] >= claim["end_line"]:
                print("Error - start line is greater than end line")
                continue

        if assignments_to_make > 0:

            print("Added", "{}?annotationTarget={}".format(args.ext_url, str(claim["_id"])))
            num_assignments = assignments_to_make

            try:
                assert not (claim["annotation_count"] > 0 and sum(1 if "keep" in a and a['keep'] else 0 for a in  ds.get_annotations_from_claim(claim)) >= claim["target_annotation_count"] )
            except AssertionError as e:
                print(claim)
                print(ds.get_annotations_from_claim(claim))
                raise e

            if dry_run:
                continue

            exp_days = 5

            response = client.create_hit(Title="Wikipedia evidence finding: select sentences that verify a claim",
                                         Reward="0.12",
                                         MaxAssignments=num_assignments,
                                         AssignmentDurationInSeconds= 25 * 60,
                                         LifetimeInSeconds= exp_days * 24 * 60 * 60,
                                         Description="Select evidence from the Wikipedia page that can be used to support or refute a simple claim.",
                                         Question=ExternalQuestion(
                                             external_url="{}?annotationTarget={}".format(args.ext_url, str(claim["_id"])),
                                             frame_height=0).get_as_xml(),
                                         QualificationRequirements=[{
                                                'QualificationTypeId': '00000000000000000071',
                                                'Comparator': 'In',
                                                'LocaleValues': [
                                                    {
                                                        'Country': 'US'
                                                    },
                                                    {
                                                        'Country': 'GB'
                                                    },
                                                    {
                                                        "Country": 'CA'
                                                    }
                                                ],
                                                'RequiredToPreview': False,
                                                'ActionsGuarded': 'Accept'
                                            },
                                             {
                                                 'QualificationTypeId': '00000000000000000040',
                                                 'Comparator': 'GreaterThanOrEqualTo',
                                                 'IntegerValues': [1000],
                                                 'RequiredToPreview': False,
                                                 'ActionsGuarded': 'Accept'
                                             },

                                             {
                                                 'QualificationTypeId': '000000000000000000L0',
                                                 'Comparator': 'GreaterThanOrEqualTo',
                                                 'IntegerValues': [92],
                                                 'RequiredToPreview': False,
                                                 'ActionsGuarded': 'Accept'
                                             },

                                            {
                                                "QualificationTypeId": pending_qualification,
                                                 "Comparator": "DoesNotExist",
                                                 'RequiredToPreview': False,
                                                 'ActionsGuarded': 'Accept'
                                            },
                                            {
                                                 "QualificationTypeId": timeout_qualification,
                                                 "Comparator": "DoesNotExist",
                                                 'RequiredToPreview': True,
                                                 'ActionsGuarded': 'PreviewAndAccept'
                                            },
                                            {
                                                 "QualificationTypeId": main_qualification,
                                                 "Comparator": "GreaterThanOrEqualTo",
                                                 'ActionsGuarded': 'Accept',
                                                 'IntegerValues': [75]
                                            }
                                         ])

            hit_type_id = response['HIT']['HITTypeId']
            hit_id = response['HIT']['HITId']
            print("\nCreated HIT: {}".format(hit_id))

            ds.claims.update({"_id": claim["_id"]},
                             {"$push": {
                                 "hits": {
                                     "id": hit_id,
                                     "type": hit_type_id,
                                     "created": datetime.now(),
                                     "expires": datetime.now() + timedelta(hours=exp_days*24),
                                     "expected_num": num_assignments,
                                     "sandbox": "sandbox" in endpoint_url},
                                 "active_hits": hit_id}
                             })
            added += 1
            if added>10000:
               break
            print(added)
    cursor.close()


