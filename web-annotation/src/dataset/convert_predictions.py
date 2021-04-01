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

if __name__ == "__main__":
    with open("data/boolq.train.jsonl") as file, open("data/boolq.train.pred=title.jsonl", "w+") as ofile:
        for line in file:
            instance = json.loads(line)
            instance["predicted_documents"] = [normalize_text_to_title(instance["title"])]
            ofile.write(json.dumps(instance)+"\n")
