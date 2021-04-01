import json
from collections import defaultdict
from itertools import zip_longest

from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data_service import DataService

from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return json.JSONEncoder.default(self, o)

def offset(annotation):
    if "start_line" in annotation:
        return annotation["start_line"]
    return 0


def get_verdict(verdict):
    if verdict == 1:
        return 'true'
    elif verdict == -1:
        return 'false'
    else:
        return 'neutral'


def get_new_evidence(annotation, claim):
    if annotation["correction_type"] is None:
        if offset(claim) == 0 and annotation['evidence'].keys() == annotation["original_evidence"].keys():
            corrections = annotation["original_evidence"]
        else:
            corrections = []
            print("Failed corrections")
    else:
        corrections = annotation["correction_type"].lower().split()



    evidence = {int(k):v for k,v in annotation["original_evidence"].items()}
    for correction in corrections:
        if correction[0] == "+":
            correction = correction.replace("+","")
            label = None
            if "t" in correction:
                label = +1
                correction = int(correction.replace("t",""))

            elif "f" in correction:
                label = -1
                correction = int(correction.replace("f",""))

            if label is not None:
                evidence[correction-offset(claim)] = label

        elif correction[0] == "-":
            correction = int(correction.replace("-",""))
            evidence[correction-offset(claim)] = 0

    return evidence


def maybe_get_new_evidence(annotation,claim):
    if "original_evidence" in annotation:
        return get_new_evidence(annotation,claim).items()
    else:
        return annotation["evidence"].items()


def get_evidence(annotation, claim):
    ret = []
    for line, verdict in maybe_get_new_evidence(annotation,claim):
        line = int(line)

        ev = {"line": line + offset(claim), "count":1 ,"verdict": get_verdict(verdict),"annotator": obfuscate[annotation["worker_id"]]}
        if obfuscate[annotation['worker_id']] is None:
            print(annotation["worker_id"], line)
        ret.append(ev)

    return ret

if __name__ == "__main__":

    fever_db = FEVERDocumentDatabase("newdb/new.db")
    ds = DataService()

    worker_ids = dict()
    for worker in ds.workers.find():
        if "sessions" in worker:
            for session in worker["sessions"]:
                worker_ids[session] = worker['worker_id']

    worker_obfuscated = dict()
    worker_list = list(set(worker_ids.values()))

    obfuscate = defaultdict(lambda : None)
    for session, wid in worker_ids.items():
        worker_obfuscated[session] = worker_list.index(wid)
        obfuscate[wid] = worker_list.index(wid)


    all_annotations = list(tqdm(ds.annotations.find({})))
    all_claims = [anno["claim"] for anno in all_annotations]

    all_claims_cursor = list(ds.claims.find({"_id": {"$in": all_claims}}))

    claims_dict = {}
    for claim in all_claims_cursor:
        claims_dict[claim["_id"]] = claim


    dataset = {}
    for idx, annotation in enumerate(tqdm(all_annotations)):
        if "keep" in annotation and not annotation["keep"]:
            continue


        claim = claims_dict[annotation["claim"]]

        if "dirty" in annotation or "dirty" in claim:
            continue

        id = claim["file"] + "_" + str(claim["idx"])
        if id in dataset:
            instance = dataset[id]
        else:
            instance = {
                "source": {
                    "source_file": claim["file"],
                    "source_idx": claim["idx"],
                    "source_question": claim["question"]
                },
                "evidence": [{
                    "entity": claim["entity"],
                    "annotations": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
                    "metadata": defaultdict(list)
                }],
                "metadata": {
                    "hits": []
                },
                "claim": claim["claim_text"]

            }
            dataset[id] = instance

        for evidence in get_evidence(annotation, claim):
            instance["evidence"][0]["annotations"][claim["wiki"]][evidence["line"]][evidence["verdict"]] += [evidence["annotator"]]


        page = fever_db.get_doc_lines(claim["wiki"])
        to_del = []
        for line_id, verdicts in instance["evidence"][0]["annotations"][claim["wiki"]].items():

            if line_id >= len(page):
                print(f"{line_id} exceeds length for page")
                to_del.append(line_id)
            elif len(page[line_id].split("\t")) <=1 or not len(page[line_id].split("\t")[1].strip()):
                assert "true" not in verdicts
                assert "false" not in verdicts
                to_del.append(line_id)

        for line_id in to_del:
            del instance["evidence"][0]["annotations"][claim["wiki"]][line_id]

        instance["evidence"][0]["metadata"][claim["wiki"]].append({
            "top_level_prediction":annotation["submit_type"],
            "relevant": annotation["relevant"],
            "certainty": annotation["certain"],
            "unsure": annotation["unsure"],
            "start_line": claim["start_line"] if "start_line" in claim else None,
            "end_line": claim["end_line"] if "end_line" in claim else None,
            "annotation_time": annotation["annotation_time"],
            "annotator": obfuscate[annotation["worker_id"]]
        })

        if obfuscate[annotation["worker_id"]] is None:
            print("IS None " + str(annotation['worker_id']))

        instance["metadata"]["hits"].extend(claim["hits"])


    with open("generated_dataset_v0.6_2021_03_31.jsonl","w+") as f:
        for item in dataset.values():
            f.write(json.dumps(item, default=str)+"\n")
            continue

            y = list(item["evidence"][0]["metadata"].keys())[0]
            x = item["evidence"][0]['annotations'][y]
            a = fever_db.get_doc_lines(y)

            try:
                assert len(x) == len(a)

                #print(list(zip_longest(fever_db.get_doc_lines(a, list(list(x['annotations'].values())[0].values())))))
                for s,r in zip_longest(a,x.values()):
                    print(s, r)

                print()
                print()
                print()
            except:
                print(y)
                print(x)
                print(a)