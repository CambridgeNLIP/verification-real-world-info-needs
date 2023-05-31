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

for idx, claim in enumerate(ds.claims.find({"series":"boolq-sample-100claims-pilot-series2-variant-1"})):
    print("Claim")
    annotations = ds.annotations.find({"claim":claim["_id"]})
    print("Found annotations")
    for anidx, annotation in enumerate(annotations):
        out_text = "Claim: {}\nPage: {}\nDate: {}\nWorkerId: {}\nAnnotationTime: {}\nDictionary Expansions: {}\nNot Enough Info: {}".format(
            claim["claim_text"],
            claim["page"],
            annotation["created"],
            annotation["username"],
            annotation["annotation_time"],
            annotation["num_sentences_visited"],
            annotation["submit_type"]
        )

        print(annotation)
        times.append(annotation['annotation_time'])



        try:
            lines = get_wiki_clean_db(db, claim["page"], redirects)["text"].split("\n")
        except:
            continue

        ev = []

        for s_idx in annotation["support_sents"]:
            ev.append(lines[s_idx])

            annotation_selections = annotation["selections"][str(s_idx)]

            for linked_page, selections in annotation_selections.items():

                if any(selections.values()):
                    extra_ids = [int(s[0]) for s in selections.items() if s[1]]
                    extra_lines = get_wiki_clean_db(db, linked_page, redirects)["text"].split("\n")
                    append_lines = "\n\t\t".join([extra_lines[id].split("\t")[1] for id in extra_ids])

                    ev.append("\tJOINT Evidence with {} {} \n\t\t{}".format(linked_page, [int(s[0]) for s in selections.items() if s[1]], append_lines))


        if len(ev):
            out_text += "\n\nSupported Evidence:\n\t"
            out_text += "\n\t".join([e.split('\t',maxsplit=1)[1] for e in ev])


        ev = []
        for s_idx in annotation["refute_sents"]:
            ev.append(lines[s_idx])
            annotation_selections = annotation["selections"][str(s_idx)]

            for linked_page, selections in annotation_selections.items():

                if any(selections.values()):
                    extra_ids = [int(s[0]) for s in selections.items() if s[1]]
                    extra_lines = get_wiki_clean_db(db, linked_page, redirects)["text"].split("\n")
                    append_lines = "\n\t\t".join([extra_lines[id].split("\t")[1] for id in extra_ids])
                    ev.append("\tJOINT Evidence with {} {} \n\t\t{}".format(linked_page, [int(s[0]) for s in selections.items() if s[1]], append_lines))



        if len(ev):
            out_text += "\n\nRefuted Evidence:\n\t"
            out_text += "\n\t".join([e.split('\t',maxsplit=1)[1] for e in ev])

        with open("pilot/series1/{}-{}.txt".format(claim['_id'],anidx),"w+") as f:
            f.write(out_text)

    with open("pilot/series1/timings.txt","w+") as f:
        f.write('\n'.join(str(t) for t in times))
