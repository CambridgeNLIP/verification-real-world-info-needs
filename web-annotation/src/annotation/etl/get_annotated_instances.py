import json

from tqdm import tqdm
from annotation.data_service import DataService

if __name__ == "__main__":
    ds = DataService()
    annotations = ds.annotations.find()

    claims = set()
    for annotation in annotations:
        claims.add(annotation["claim"])

    collected = set()
    original = ds.claims.find({"_id":{"$in":list(claims)}})
    for item in tqdm(original):
        collected.add((item["file"],item["idx"]))

    with open("pre_annotated.json","w+") as f:
        json.dump(list(collected),f)