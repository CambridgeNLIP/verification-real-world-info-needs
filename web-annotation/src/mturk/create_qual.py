from datetime import datetime

import boto3
from boto.mturk.question import ExternalQuestion



def create_qualification_type(mturk, name):
    paginator = mturk.get_paginator('list_qualification_types')
    found = False
    found_id = None

    for quals in paginator.paginate(MustBeRequestable=False, MustBeOwnedByCaller=True):
        for obj in quals["QualificationTypes"]:

            if obj["Name"] == name:
                found = True
                found_id = obj["QualificationTypeId"]
                break

    if not found:
        response = mturk.create_qualification_type(Name=name,
                                    Keywords="claim,wikipedia,evidence,search,true,false,finding,natural,question,ai",
                                    Description="Pass a qualification test to show you can find information from a Wikipedia page that supports or refutes a claim.",
                                    QualificationTypeStatus="Active",
                                    RetryDelayInSeconds=60*60*24)
        return response["QualificationType"]["QualificationTypeId"]

    return found_id


if __name__ == "__main__":
    endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'

    # Uncomment this line to use in production
    # endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'


    mturk = boto3.client(
        'mturk',
        endpoint_url=endpoint_url, )

    print(create_qualification_type(mturk, "Wikipedia Evidence Finding"))
    print(create_qualification_type(mturk, "Wikipedia Evidence Finding Awaiting Qualification"))