import json
import logging
import os
import re
import sys
import time
import xml
from argparse import ArgumentParser
from bz2 import BZ2File
from multiprocessing import Process
from threading import Thread

import spacy
from drqa.retriever.utils import normalize

from dataset.construction.article_queue import ArticleReadingQueue
from dataset.construction.file_queue import FileQueue
from dataset.construction.read_wiki_full import join_titles
from dataset.reader.cleaning import simple_clean, post_clean, fix_quotes, fix_header
from dataset.reader.wiki_reader import WikiReader

logger = logging.getLogger(__name__)
shutdown = False

def display():
    while True:
        logger.debug("Queue sizes {0} {1} {2}".format(arq.redirect_queue.qsize(), arq.article_queue.qsize(), reader.num_articles))
        time.sleep(1)

def setup_logger():
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)



def get_link_text(match):
    return match.group(1) if "|" not in match.group(1) else match.group(1).split("|")[1]

def clean(text):
    return re.sub(r'\[\[([^\]]+)\]\]',get_link_text,text),[]


def skip(page_title):
    pt = normalize(page_title).lower()
    return any(pt.startswith(a) for a in ["list of","bibliograpy of"])

def process_article():

    while not (shutdown and arq.article_queue.empty()):
        page_title,source = arq.article_queue.get()
        if skip(page_title):
            continue
        text = fix_header(fix_quotes(post_clean(simple_clean(source))))

        sections = []
        section = []
        title = ""
        title_2 = ""
        title_3 = ""
        for section_idx, line in enumerate(text.split("\n")):
            if line.startswith("<h2>") or line.startswith("<h3>") or line.startswith("<h4>"):
                if len(section) and any(len(s) for s in section):

                    sections.append((join_titles(title,title_2,title_3),section))
                    section = []

                if line.startswith("<h2>"):
                    title = line
                    title_2 = ""
                    title_3 = ""

                if line.startswith("<h3>"):
                    title_2 = line
                    title_3 = ""

                if line.startswith("<h4>"):
                    title_3 = line

            else:
                section.append(line)

        if len(section) and any(len(s) for s in section):
            sections.append((join_titles(title,title_2,title_3),section))

        for title,section in filter(lambda sec: not any([sec[0].lower().strip().startswith(s)
                                                     for s in ["see also",
                                                               "notes and references",
                                                               "references",
                                                               "external links",
                                                               "further reading",
                                                               "bibliography"]]),
                                    sections):

            section_text, section_links = clean(("\n".join(section)).strip())
            title, _ = clean(title)
            doc = nlp(section_text.strip())

            sents = []
            try:
                for s in doc.sents:
                    if len(sents) > 0:
                        if len(str(sents[-1]).strip()) and str(sents[-1]).strip()[-1] != ".":
                            sents[-1] += str(s)
                            continue
                    sents.append(str(s))

                all_lines = "\n".join(sents)
                all_lines = re.sub(r'(\\n){3,}','\n\n',all_lines.strip())
                out_text = ""
                for idx,s in enumerate(all_lines.split("\n")):
                    out_text += str(idx) + "\t" + s.strip() + "\n"

                fq.enqueue(json.dumps({"page": page_title ,"section":str(title).replace("=",""), "lines":out_text}))
            except:
                logger.critical("Sentencizer failed on this page / section: {}".format(section_text.strip()) )


def process_redirect():
    while not (shutdown and arq.redirect_queue.empty()):
        line = arq.redirect_queue.get()
        dest_file.write(line[0] + "\t" + line[1] + "\n")

def write_out():
    while not (shutdown and fq.q.empty()):
        line = fq.q.get()
        out_file.write(line+"\n")

if __name__ == "__main__":
    setup_logger()
    
    logger.info("Read Wikipedia")

    parser = ArgumentParser()
    parser.add_argument("wiki", help="wiki dump file .xml.bz2")
    parser.add_argument("redirects", help="redirects file .txt")
    parser.add_argument("out", help="final file .txt")
    args = parser.parse_args()

    logger.info("Load spacy")
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))

    logger.info("Read wiki file: {}".format(args.wiki))
    wiki = BZ2File(args.wiki)
    dest_file = open(os.path.join(args.redirects),"w+")

    out_file = open(os.path.join(args.out),"w+")

    arq = ArticleReadingQueue()
    fq = FileQueue()

    reader = WikiReader(lambda ns: ns == 0, arq.enqueue_article, arq.enqueue_redirect)


    thread = Thread(target=display, args=())
    thread.daemon = True  # Daemonize thread
    thread.start()  # Start the execution

    for _ in range(15):
        t = Process(target=process_article)
        t.start()

    t2 = Thread(target=process_redirect)
    t2.start()

    t3 = Thread(target=write_out)
    t3.start()


    try:
        xml.sax.parse(wiki, reader)
    except Exception as e:
        logger.error(e)
