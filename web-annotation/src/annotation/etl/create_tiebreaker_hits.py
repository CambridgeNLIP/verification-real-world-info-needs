import os, boto3
from datetime import datetime, timedelta

from annotation.annotation_service import qualify_worker
from annotation.data_service import DataService
from mturk.create_qual import create_qualification_type

if __name__ == "__main__":
    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )

    super_qualification = create_qualification_type(mturk, "Wikipedia Evidence Finding: Super Annotators")
    timeout_qualification = create_qualification_type(mturk, "Wikipedia Evidence Finding: Soft Block [Timeout]")
    pending_qualification = create_qualification_type(mturk, "Wikipedia Evidence Finding: Soft Block [Awaiting Review]")
    main_qualification = create_qualification_type(mturk, "Wikipedia Evidence Finding: Quiz 2")

    ds = DataService()

    from boto.mturk.question import ExternalQuestion

    for _ in range(11616):
        exp_days = 10
        response = mturk.create_hit(Title="Wikipedia evidence tiebreaker: select sentences that verify a claim",
                                 Reward="0.12",
                                 MaxAssignments=1,
                                 AssignmentDurationInSeconds=25 * 60,
                                 LifetimeInSeconds=exp_days * 24 * 60 * 60,
                                 Description="[Tiebreaker] Select evidence from the Wikipedia page that can be used to support or refute a simple claim.",
                                 Question=ExternalQuestion(
                                     external_url="https://annotation-live.jamesthorne.co.uk/mturk/tiebreaker/12",
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
                                         'QualificationTypeId': super_qualification,
                                         "Comparator": "Exists",
                                         'RequiredToPreview': True
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
        print("Created HIT: {}".format(hit_id))

        ds.tie_breaker_hits.insert({
                 "hit": hit_id,
                 "type": hit_type_id,
                 "created": datetime.now(),
                 "expires": datetime.now() + timedelta(hours=exp_days * 24),
                 "expected_num": 5,
                 "sandbox": "sandbox" in endpoint_url
        })

"""
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

"""
