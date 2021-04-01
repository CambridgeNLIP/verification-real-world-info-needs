from annotation.data_service import DataService
from csv import DictWriter
ds = DataService()

records = []


def review(annotation):
    if "keep" not in annotation:
        return "NOT REVIEWED"
    elif "correction_type" in annotation and (annotation["correction_type"] is not None and len(annotation["correction_type"].strip())):

        return "CORRECTED"
    elif "keep" in annotation and annotation["keep"]:
        if "auto_accept" in annotation:
            return "AUTO ACCEPT"
        return "ACCEPT"

    return "REJECT"


def variant(annotation):
    if annotation["created"].month >= 5:
        return 6
    else:
        return annotation["variant"]


def corrected(annotation):
    return "correction_type" in annotation and annotation["correction_type"]


for annotation in ds.annotations.find():
    record = {
        "_id": str(annotation["_id"]),
        "claim": str(annotation["claim"]),
        "created": str(annotation["created"]),
        "assignment": str(annotation["assignment_id"]),
        "worker": str(annotation["worker_id"]),
        "annotation_time": annotation["annotation_time"],
        "submit_type": annotation["submit_type"],
        "keep": review(annotation),
        "variant": variant(annotation)
    }
    records.append(record)

with open("records.csv", "w+") as f:
    writer = DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)


