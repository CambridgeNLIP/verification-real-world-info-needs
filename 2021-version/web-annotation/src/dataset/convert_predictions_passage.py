import json
import unicodedata


def normalize(text):
    return unicodedata.normalize('NFD', text)
def normalize_text_to_title(text):
    return normalize(text.strip()\
        .replace("(","-LRB-")\
        .replace(")","-RRB-")\
        .replace(" ","_")\
        .replace(":","-COLON-")\
        .replace("[","-LSB-")\
        .replace("]","-RSB-"))

with open("data/boolq.train.jsonl") as file, open("data/boolq.train.pred=title,claim=passage.jsonl", "w+") as ofile:
    for line in file:
        instance = json.loads(line)
        instance["claim_text"] = instance["claim"]
        instance["claim"] = instance["passage"]
        instance["predicted_documents"] = [normalize_text_to_title(instance["title"])]
        ofile.write(json.dumps(instance)+"\n")
