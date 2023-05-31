import argparse
import os

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data.redirect_db import RedirectDatabase
from annotation.data_service import DataService
from util.wiki import get_wiki_clean_db

parser = argparse.ArgumentParser()
parser.add_argument("--db", type=str, required=True)
args = parser.parse_args()
ds = DataService()
db = FEVERDocumentDatabase(args.db)

redirects_db_loc = os.getenv("REDIRECTS_DB", "data/redirects.db")
redirects = RedirectDatabase(redirects_db_loc)

times = []


for idx, claim in enumerate(ds.claims.find({"series":"boolq-sample-100claims-pilot-series2-variant-2"})):
    annotations = ds.annotations.find({"claim":claim["_id"]})

    for anidx, annotation in enumerate(annotations):
        out_text = "Claim: {}\nPage: {}\nDate: {}\nWorkerId: {}\nAnnotationTime {}\nNot Enough Info: {}".format(
            claim["claim_text"],
            claim["page"],
            annotation["created"],
            annotation["username"],
            annotation["annotation_time"],
            annotation["submit_type"]
        )

        times.append(annotation['annotation_time'])



        try:
            lines = get_wiki_clean_db(db, claim["page"], redirects)["text"].split("\n")
        except:
            continue

        ev = []

        for idx in annotation["support_sents"]:
            ev.append(lines[idx])
        if len(ev):
            out_text += "\n\nSupported Evidence:\n\t"
            out_text += "\n\t".join([e.split('\t')[1] for e in ev])

        ev = []
        for idx in annotation["partial_support_sents"]:
            ev.append(lines[idx])
        if len(ev):
            out_text += "\n\nPartial Support Evidence:\n\t"
            out_text += "\n\t".join([e.split('\t')[1] for e in ev])

        ev = []
        for idx in annotation["partial_refute_sents"]:
            ev.append(lines[idx])
            if len(ev):
                out_text += "\n\nPartial Refute Evidence:\n\t"
                out_text += "\n\t".join([e.split('\t')[1] for e in ev])

        ev = []
        for idx in annotation["refute_sents"]:
            ev.append(lines[idx])

        if len(ev):
            out_text += "\n\nRefuted Evidence:\n\t"
            out_text += "\n\t".join([e.split('\t')[1] for e in ev])

        with open("pilot/series2/{}-{}.txt".format(claim['_id'],anidx),"w+") as f:
            f.write(out_text)

    with open("pilot/series2/timings.txt","w+") as f:
        f.write('\n'.join(str(t) for t in times))
