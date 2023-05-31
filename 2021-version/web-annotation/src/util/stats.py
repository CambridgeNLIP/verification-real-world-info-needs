from argparse import ArgumentParser
from tqdm import tqdm
from annotation.data.fever_db import FEVERDocumentDatabase
from scipy.stats import describe
import numpy as np

parser = ArgumentParser()
parser.add_argument("--db", required=True)
args = parser.parse_args()
db = FEVERDocumentDatabase(args.db)
ids = db.get_ids()

xlines = []
paras = []
ttoks = []

def percentiles(obs):
    return [np.percentile(obs, i) for i in [10,20,30,40,50,60,70,80,90]]

def iqr(obs):
    return [np.percentile(obs, i) for i in [1,5,25,50,75,95,100]]

for id in tqdm(ids):
    title,lines = db.get_doc_lines_and_tile(id[0])


    if title.lower().startswith("list of"):
        continue
    non_empty_lines = len([l for l in lines if len(l.split("\t")) > 1 and len(l.split("\t")[1].strip())])

    in_text = True

    mode_switch = 1
    toks = 0
    for line in lines:
        if len(line.split("\t")) >1 and line.split("\t")[1].strip() == "":
            in_text = False
        else:
            if not in_text:
                mode_switch+=1
            in_text = True

            if len(line.split("\t")) > 1:
                toks += len([t for t in line.split("\t")[1].split(" ") if t != ""])
    ttoks.append(toks)
    xlines.append(non_empty_lines)
    paras.append(mode_switch)

print(describe(xlines))
print(percentiles(xlines))
print(iqr(xlines))
print("---")
print(describe(paras))
print(percentiles(paras))
print(iqr(paras))
print("---")
print(describe(ttoks))
print(percentiles(ttoks))
print(iqr(ttoks))
print("---")
