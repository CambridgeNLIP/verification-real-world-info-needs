import datetime
import logging
import json
import os
import requests
from argparse import ArgumentParser

from tqdm import tqdm

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data.redirect_db import RedirectDatabase
from annotation.data_service import DataService


logger = logging.getLogger(__name__)

logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"),
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--in-file", type=str, required=True)
    parser.add_argument("--out-file", type=str, required=True)
    args = parser.parse_args()

    with open(args.in_file) as infile, open(args.out_file,"w+") as outfile:
        for idx, row in enumerate(tqdm(infile)):
            row = json.loads(row)

            resp = requests.post("http://localhost:5000/predict",json={"instances":[{"id":1,"claim":row["claim"]}]})
            if resp.status_code == 200:

                row["wiki"] = resp.json()["data"]["predictions"][0]["request_instance"]["wiki_results"]
                outfile.write(json.dumps(row)+"\n")
