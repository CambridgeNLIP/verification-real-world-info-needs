import wikipedia
import re
import logging
from html2text import html2text, HTML2Text
from wikipedia.wikipedia import _wiki_request

logger = logging.getLogger(__name__)


def fix_header(text):
    text = re.sub(r"^====(.+)====$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^===(.+)===$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^==(.+)==$", r"<h2>\1</h2>", text, flags=re.MULTILINE)

    text = re.sub(r"^####(.+)$", r"<h4>\1</h4>", text, flags=re.MULTILINE)
    text = re.sub(r"^###(.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^##(.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)

    return text


def join_titles(*titles):
    ret = []
    for title in titles:
        found = re.sub(r'<h[1-6]>(.+)</h[1-6]>', r'\1', title).strip()
        if len(found):
            ret.append(found)
    return " &raquo; ".join(ret) if len(ret) else ""


def get_link_text(match):
    return match.group(1) if "|" not in match.group(1) else match.group(1).split("|")[1]


def clean(text):
    return re.sub(r'\[\[([^\]]+)\]\]', get_link_text, text), []


def newhtml2text(html, baseurl=None, bodywidth=None):
    h = HTML2Text(baseurl=baseurl, bodywidth=bodywidth)
    h.ignore_emphasis = True
    return h.handle(html)


def fix_paragraphs(text):
    text = newhtml2text(text, bodywidth=0)

    return text


def get_html_extract(page):
    if not getattr(page, '_content', False):
        query_params = {
            'prop': 'extracts|revisions',
            'rvprop': 'ids'
        }
        if not getattr(page, 'title', None) is None:
            query_params['titles'] = page.title
        else:
            query_params['pageids'] = page.pageid
        request = _wiki_request(query_params)
        page._content = fix_paragraphs(request['query']['pages'][page.pageid]['extract'])
        page._revision_id = request['query']['pages'][page.pageid]['revisions'][0]['revid']
        page._parent_id = request['query']['pages'][page.pageid]['revisions'][0]['parentid']

    return page._content


def get_wiki_sections(splitter, page_name):
    page = wikipedia.page(title=page_name,auto_suggest=False)
    html = get_html_extract(page)
    header_fix = fix_header(page.content)

    sections = []
    section = []
    title = ""
    title_2 = ""
    title_3 = ""

    ret = []
    for section_idx, line in enumerate(header_fix.split("\n")):
        if line.startswith("<h2>") or line.startswith("<h3>") or line.startswith("<h4>"):
            if len(section) and any(len(s) for s in section):
                sections.append((join_titles(title, title_2, title_3), section))
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
        sections.append((join_titles(title, title_2, title_3), section))

    for title, section in filter(lambda sec: not any([sec[0].lower().strip().startswith(s)
                                                      for s in ["see also",
                                                                "notes and references",
                                                                "references",
                                                                "external links",
                                                                "further reading",
                                                                "bibliography"]]),
                                 sections):
        title, _ = clean(title)
        sents = []
        for section_text in section:
            section_text = section_text.replace(".)", ").")
            section_text = section_text.replace(".\"", "\".")
            section_text = section_text.replace(".'", "'.")

            section_text = re.sub(r'(^[\* ]+$)', '', section_text)
            doc = splitter.segment(section_text.strip())

            for s in doc:
                if len(sents) > 0:
                    if len(str(sents[-1]).strip()) and str(sents[-1]).strip()[-1] != ".":
                        sents[-1] += str(s)
                        continue

                if len(s):
                    sents.append(str(s))

            if len(sents):
                sents.append("")

        if len(sents):
            all_lines = "\n".join(sents)
            all_lines = re.sub(r'(\\n){3,}', '\n\n', all_lines.strip())
            out_text = ""
            for idx, s in enumerate(all_lines.split("\n")):
                out_text += str(idx) + "\t" + s.strip() + "\n"
            ret.append((title, all_lines.split("\n")))

    return ret