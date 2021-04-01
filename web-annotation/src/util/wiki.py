# Copyright 2018 Amazon Research Cambridge
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import os

from tqdm import tqdm

from util.untokenize import untokenize

logger = logging.getLogger(__name__)

def get_wiki_clean_db(db, name,redirects):
    """
    First tries to get the entity from S3 without modifying the entity name
    If the key is not found, it will apply camel-case capitalisation and look up entity in redirects.txt

    :param name: The name of the entity
    :return: a dictionary containing the body of the intro for the entity
             and the canonical entity name after resolving it
    """
    entry = get_wiki_entry(db, name, redirects)
    if entry is None:
        print(name, "not found")
        return None

    text = entry["text"]
    # Replace problematic dictionary entries (e.g. claim #195054)
    # for some reason, there are line breaks between the linked text and the URL of the entry
    text = text.replace('\n\t', '\t')

    out = []
    for line in text.split("\n"):
        if len(line.split("\t")) > 1:
            linebits = line.split("\t")
            linebits[1] = untokenize(linebits[1])

            out.append("\t".join(linebits))
        else:
            out.append(line)

    entry["text"] = "\n".join(out)
    return entry

def get_redirects():
    redirs = os.path.join("data", "redirect.txt")
    rd = dict()
    for line in tqdm(open(redirs, encoding='utf-8')):
        bits = line.strip().split("\t")
        if len(bits) == 2:
            frm, to = bits
            rd[frm] = to
    return rd


def recursive_redirect_lookup(redirects_list, word):
    try:
        return recursive_redirect_lookup(redirects_list, redirects_list.get_doc_lines(word))
    except:
        return word

def get_wiki_entry(db, name,redirects):
    name = name.strip().split("#")[0]

    try:
        body = db.get_doc_lines(name.replace(" ","_").replace("(","-LRB-").replace(")","-RRB-"))
        return {"text": "\n".join(body), "canonical_entity": name}

    except Exception as e:
        try:
            return get_wiki_entry(db, recursive_redirect_lookup(redirects, redirects[name]),redirects)
        except RecursionError:
            logger.error("Couldn't resolve {0} from dictionary: recursive redirect loop".format(name))
            return None
        except:
            if name[0].islower():
                try:
                    return get_wiki_entry(db, [name[0].upper()]+name[1:], redirects)
                except:
                    try:
                        return get_wiki_entry(db, name.title(), redirects)
                    except:
                        pass

            words = (name[0].upper() + name[1:]).split()


            for idx,word in enumerate(words):
                if idx == 0:
                    continue

                try:
                    return get_wiki_entry(db, " ".join(words[:idx]+[word.lower()]+words[idx+1:]), redirects)
                except:
                    pass


            logger.error("Could not resolve {0} from dictionary because it doesnt exist".format(name))
            return None
