import json

from numpy import mean, std

num_pages = []
with open("boolq.wikis.jsonl") as f:
    for idx, line in enumerate(f):
        with open("wiki/{}.txt".format(idx),"w+") as outf:
            t = json.loads(line)
            print(t)
            outf.write(t["claim"]+"\n\n")
            outf.write("\n".join(t["wiki"]))
            num_pages.append(len(t["wiki"]))

print(mean(num_pages), std(num_pages))