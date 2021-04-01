import json
import os

import pysbd
import multiprocessing
from argparse import ArgumentParser
from collections import defaultdict

import logging
from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data.redirect_db import RedirectDatabase
from dataset.construction.get_wiki import get_wiki_sections
from util.wiki import get_wiki_clean_db

logger = logging.getLogger(__name__)
def download_page(title):
    try:
        return (title, get_wiki_sections(splitter, title))
    except Exception as e:

        if hasattr(e, 'message'):
            message = e.message
        else:
            message="exception"

        print(e)
        return (title, None, message)

if __name__ == "__main__":
    generated = []
    parser = ArgumentParser()
    parser.add_argument("--in-files", nargs="+", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    parser.add_argument("--db", type=str, default="data/fever-sections.db")

    args = parser.parse_args()
    db = FEVERDocumentDatabase(args.db)

    claims = defaultdict(set)
    questions = defaultdict(list)

    instances = []
    titles = set()

    logger.info("installing redirects")
    redirects_db_loc = os.getenv("REDIRECTS_DB","redirects.db")

    try:
        if not os.path.exists(redirects_db_loc):
            redirects = RedirectDatabase(redirects_db_loc)
            redirects.install(os.getenv("REDIRECTS_PATH", "data/redirects.txt"))
    except:
        pass

    redirects = RedirectDatabase(redirects_db_loc)


    splitter = pysbd.Segmenter(language="en",clean=False)

    for file in args.in_files:
        with open(file) as infile:
            for idx, row in tqdm(enumerate(infile)):
                row = json.loads(row)
                title = row['title']
                try:
                    titles.add(redirects.get_doc_lines(row["title"]))
                except:
                    titles.add(row["title"])

                row["filename"] = file
                row["read_idx"] = idx
                instances.append(row)
                claims[row["question"]].add(row["claim"])
                questions[row["question"]].append(row)

    with open(args.out_file,"w+") as f:
        pool = multiprocessing.Pool()
        for downloaded in tqdm(pool.imap(download_page, titles),total=len(titles)):
            if downloaded[1] is not None:
                f.write(json.dumps({
                    "title": downloaded[0],
                    "content": [{"section": sec[0], "sentences": sec[1]} for sec in downloaded[1]],
                }) + "\n")

            else:
                f.write(json.dumps({
                    "title": downloaded[0],
                    "error": downloaded[2],
                }) + "\n")






