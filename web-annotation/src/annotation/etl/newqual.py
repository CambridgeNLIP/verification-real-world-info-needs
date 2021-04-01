from collections import defaultdict
from datetime import datetime, timedelta
import logging
import os
from argparse import ArgumentParser
from typing import List
import sys


from annotation.data_service import DataService
from mturk.create_qual import create_qualification_type


logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


import boto3
from boto.mturk.question import ExternalQuestion

endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

# Uncomment this line to use in production
#endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'

client = boto3.client(
    'mturk',
    endpoint_url=endpoint_url,)




if __name__ == "__main__":
    main_qualification = create_qualification_type(client, "quiz test 173-new-h")
    print(main_qualification)

    test = " ".join(open("qualify.xml").readlines()).strip()

    print(test)

    answers = " ".join(open("qualify_answers.xml").readlines())

    client.update_qualification_type(QualificationTypeId=main_qualification,
                                     Test=test,
                                     AnswerKey=answers,
                                     TestDurationInSeconds=60*10,
                                     RetryDelayInSeconds=60*0)
    xp = client.get_paginator('list_qualification_requests')
    for qual_req in xp.paginate(
        QualificationTypeId=main_qualification,

    ):
        print(qual_req)

    response = client.create_hit(Title="Wikipedia evidence finding: select sentences that verify a claim H",
                                 Reward="0.12",
                                 MaxAssignments=10,
                                 AssignmentDurationInSeconds=25 * 60,
                                 LifetimeInSeconds=8 * 60 * 60,
                                 Description="Select evidence from the Wikipedia page that can be used to support or refute a simple claim.",
                                 Question=ExternalQuestion(
                                     external_url="{}".format("https://jamesthorne.co.uk"),
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
                                         "QualificationTypeId": main_qualification,
                                         "Comparator": "GreaterThanOrEqualTo",
                                         'ActionsGuarded': 'Accept',
                                         'IntegerValues': [90]
                                     },
                                 ]
                                 )