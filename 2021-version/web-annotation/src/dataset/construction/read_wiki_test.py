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



if __name__ == "__main__":
    setup_logger()
    logger.info("Load spacy")
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))
    logger.info("Parsing")
    source = """
    {{Use mdy dates|date=May 2020}}
{{about|the IRT Broadway–Seventh Avenue Line station|the IND Eighth Avenue Line station|34th Street–Penn Station (Eighth Avenue)|other uses|34th Street station (disambiguation){{!}}34th Street station}}
{{Short description|New York City Subway station in Manhattan}}
{{Infobox NYCS
| name = 34 Street–Penn Station
| image = NYC subway Pennsylvania Pano.JPG 
| image_caption = Northbound local platform
| accessible = yes
| wifi = yes
| address = West 34th Street &amp; Seventh Avenue<br>New York, NY 10001
| borough = [[Manhattan]]
| locale = [[midtown, Manhattan|Midtown]]
| coordinates = {{coord|40.751|N|73.991|W|display=inline,title}}
| lat_dir = N
| lon_dir = W
| bg_color = #E20F00
| division = IRT
| line = [[IRT Broadway–Seventh Avenue Line]]
| service = Broadway-Seventh center south
| connection = {{bus icon}} '''[[NYCT Bus]]''': {{NYC bus link|M4|M7|M20|M34 SBS|M34A SBS|Q32}}, <br>{{bus icon}} '''[[MTA Bus]]''': {{NYC bus link|BxM2}}<br>[[File:BSicon BAHN.svg|12px|alt=Railway transportation]] '''[[Amtrak]]''', '''[[Long Island Rail Road|LIRR]]''', '''[[NJT Rail]]''' (at [[Pennsylvania Station (New York City)|Penn Station]])<br> {{rint|path|18px}} '''[[Port Authority Trans-Hudson|PATH]]''': [[JSQ–33]], [[HOB–33]], [[JSQ–33 (via HOB)]] (at [[33rd Street (PATH station)|33rd Street]])
| open_date = {{start date and age|1917|06|03}}<ref name=34th>{{cite news| newspaper=New York Times| title=Three New Links of the Dual Subway System Opened| date=June 3, 1917| page=33}}</ref>
| platforms = 2 [[side platform]]s (local)<br>1 [[island platform]] (express)
| tracks = 4
| structure = Underground
| passengers = 25,968,950<ref name="2016-rider">{{NYCS const|riderref}}</ref>
| pass_year = 2018
| pass_percent = -0.3
| rank = 6
| next_north = {{NYCS next | station=Times Square–42nd Street | line=IRT Broadway–Seventh Avenue Line | service=Broadway-Seventh center south}}
| next_south = {{NYCS next | type=local | line=IRT Broadway–Seventh Avenue Line | station=28th Street | service=Broadway-Seventh center south local}}<br>{{NYCS next | type=express | line=IRT Broadway–Seventh Avenue Line | station=14th Street | service=Broadway-Seventh center south express}}
| next_north_acc = {{NYCS next | station=Times Square–42nd Street | line=IRT Broadway–Seventh Avenue Line | service=Broadway-Seventh center south}}
| next_south_acc = {{NYCS next | line=IRT Broadway–Seventh Avenue Line | station=Chambers Street | service=Broadway-Seventh south}}
| code = 318
| legend = {{NYCS infobox legend|alltimes}}{{NYCS infobox legend|nightsonly}}{{NYCS infobox legend|allexceptweekdaynights}}{{NYCS infobox legend|nightsweekends}}{{NYCS infobox legend|weekdaysonly}}{{NYCS infobox legend|weekendsnights}}
}}
'''34th Street–Penn Station''' is an express [[metro station|station]] on the [[IRT Broadway–Seventh Avenue Line]] of the [[New York City Subway]]. Located at the intersection of [[34th Street (Manhattan)|34th Street]] and [[Seventh Avenue (Manhattan)|Seventh Avenue]], it is served by the '''[[1 (New York City Subway service)|1]]''' and '''[[2 (New York City Subway service)|2]]''' trains at all times, and the '''[[3 (New York City Subway service)|3]]''' train at all times except late nights. Connections are available to the [[Long Island Rail Road|LIRR]], [[New Jersey Transit Rail Operations|NJ Transit]] and [[Amtrak]] at [[Pennsylvania Station (New York City)|Pennsylvania Station]].

==History==
34th Street–Penn Station on the IRT Broadway–Seventh Avenue Line was opened on June 3, 1917, as part of an extension of the [[Interborough Rapid Transit Company]], the dominant subway in [[Manhattan]] at the time, from Times Square–42nd Street to [[South Ferry loops (IRT Broadway–Seventh Avenue Line)|South Ferry]].<ref name=34th/> It was served by a shuttle train to Times Square until the rest of the extension opened a year later on July 1, 1918.<ref name="west side open">{{cite news| url=https://www.nytimes.com/1918/07/02/archives/open-new-subway-to-regular-traffic-first-train-on-seventh-avenue.html| title=Open new subway to regular traffic| newspaper=New York Times| accessdate=August 27, 2008}}</ref> This meant that the subway would be expanded down the Lower West Side to neighborhoods such as [[Greenwich Village]] and the western portion of the [[Financial District, Manhattan|Financial District]].

As part of this and the northern [[IRT Lexington Avenue Line]] extension, the IRT network would be radically changed from an S-shaped line connecting the eastern side of Lower Manhattan to the [[Upper West Side]] to an H-shaped network with two parallel lines, the East and West Side Lines, and a [[42nd Street Shuttle|shuttle at 42nd Street]] connecting them.<ref name="west side open"/><ref name="east side open">{{cite news| url=https://timesmachine.nytimes.com/timesmachine/1918/08/02/97011929.pdf| title=Open new subway lines to traffic; called a triumph| newspaper=New York Times| accessdate=August 27, 2008}}</ref>

On August 23, 1985, the MTA awarded a $2.24 million contract to rebuild the station and to double the width of the passageway to Penn Station. The project was scheduled to be completed in spring 1987.<ref>{{Cite news|url=|title=MTA to fund Troubled Tunnels|last=Gordy|first=Margaret|date=August 24, 1985|work=Newsday|access-date=}}</ref>

Under the 2015–2019 [[Metropolitan Transportation Authority|MTA]] Capital Plan, the station, along with thirty-two other New York City Subway stations, underwent a complete overhaul as part of the [[Enhanced Station Initiative]]. Updates included cellular service, Wi-Fi, charging stations, improved signage, and improved station lighting. Unlike other stations that were renovated under the initiative, 34th Street–Penn Station was not completely closed during construction.<ref>{{cite web|url=http://web.mta.info/nyct/procure/contracts/143675sol-3.pdf|title=Enhanced Station Initiative: CCM Pre-Proposal Conference|date=October 25, 2016|website=|publisher=[[Metropolitan Transportation Authority]]|access-date=August 11, 2017|page=8 (PDF page 15)}}</ref> In January 2018, the NYCT and Bus Committee recommended that Judlau Contracting receive the $125 million contract for the renovations of [[57th Street (IND Sixth Avenue Line)|57th]] and [[23rd Street (IND Sixth Avenue Line)|23rd Street]]s on the [[IND Sixth Avenue Line]]; [[28th Street (IRT Lexington Avenue Line)|28th Street]] on the [[IRT Lexington Avenue Line]], and 34th Street–Penn Station on the IRT Broadway–Seventh Avenue Line and [[34th Street–Penn Station (IND Eighth Avenue Line)|IND Eighth Avenue Line]].<ref>{{Cite web|url=http://web.mta.info/mta/news/books/pdf/180122_1030_Transit.pdf|title=NYCT/Bus Committee Meeting|author=[[Metropolitan Transportation Authority]]|date=January 22, 2018|page=135|accessdate=January 19, 2018|archive-url=https://web.archive.org/web/20180127073118/http://web.mta.info/mta/news/books/pdf/180122_1030_Transit.pdf|archive-date=January 27, 2018|url-status=dead}}</ref> However, the MTA Board temporarily deferred the vote for these packages after city representatives refused to vote to award the contracts.<ref>{{Cite news|url=https://www.amny.com/transit/subway-station-improvements-1.16331101|title=Controversial cosmetic subway improvement plan falters|last=Barone|first=Vincent|date=January 24, 2018|work=am New York|access-date=January 25, 2018|language=en}}</ref><ref>{{Cite news|url=https://www.nbcnewyork.com/news/local/MTA-Postpones-Plan-Modernize-Subway-Stations-Cuomo-De-Blasio-470926353.html|title=MTA Shelves Plan to Modernize Subway Stations Amid Criticism|last=Siff|first=Andrew|date=January 24, 2018|work=NBC New York|access-date=January 25, 2018|language=en}}</ref> The contract was put back for a vote in February, where it was ultimately approved.<ref>{{cite web|url=http://www.nydailynews.com/new-york/mta-votes-cuomo-1b-plan-pretty-subway-stations-article-1.3836591|title=Foes Hit Gov's Station Fix Plan|date=February 13, 2018|website=NY Daily News|access-date=February 23, 2018}}</ref> These improvements were substantially completed by May 2019.<ref>{{Cite web|url=http://web.mta.info/mta/news/books/pdf/190520_1030_Transit.pdf|title=NYCT/Bus Committee Meeting|date=May 20, 2019|accessdate=May 19, 2019|publisher=[[Metropolitan Transportation Authority]]|page=168}}</ref>

{{-|left}}

==Station layout==
{{NYCS Platform Layout Penn Station}}
[[File:NYC subway Pennsylvania 36.JPG|thumb|left|175px|Trim line tablets]]
[[File:NYC subway Pennsylvania 33.JPG|thumb|left|175px|Name on trim line]]

Like [[34th Street–Penn Station (IND Eighth Avenue Line)|34th Street–Penn Station]] on the [[IND Eighth Avenue Line]] and [[Atlantic Avenue–Barclays Center (IRT Eastern Parkway Line)|Atlantic Avenue–Barclays Center]] on the [[IRT Eastern Parkway Line]], this station has two [[side platform]]s for local service and a center [[island platform]] for express service. This is due to the expected increase in ridership and to encourage riders to switch at the next stop northbound, [[Times Square–42nd Street (IRT Broadway–Seventh Avenue Line)|Times Square–42nd Street]], as it is set up in the usual island platform manner for [[cross-platform interchange]]s.<ref name="nycsorg">{{cite web| url=http://nycsubway.org/perl/stations?6:3134| title=34th Street-Penn Station| website=NYCSubway.org| accessdate=August 27, 2008}}</ref>

There is no free transfer between this station and the station of the same name on the IND Eighth Avenue Line, despite the fact that both connect to Penn Station. The nearest transfer location is at Times Square–42nd Street with a free transfer to [[42nd Street–Port Authority Bus Terminal (IND Eighth Avenue Line)|42nd Street–Port Authority Bus Terminal]].<ref name="nycsorg"/>

{{-|left}}


{{Midtown Manhattan subway cross section}}
===Exits===
[[File:34 Street Penn Station entrance 2 vc.jpg|thumb|250px|34th Street entrance]]
34th Street–Penn Station spans three streets (32nd, 33rd, and 34th Streets) with a set of entrances/exits at all of these streets. For the purposes of this article, entrance and exit are interchangeable.<ref name=MTA-PennStation-2015>{{cite web|title=MTA Neighborhood Maps: Pennsylvania Station / Times Square|url=http://web.mta.info/maps/neighborhoods/mn/M08_PennStation_2015.pdf|website=[[Metropolitan Transportation Authority|mta.info]]|publisher=[[Metropolitan Transportation Authority]]|accessdate=December 11, 2015|date=2015}}</ref>

* {{Access icon}} 34th Street: There are four entrances directly from the intersection of 34th Street and Seventh Avenue. On the local platforms the [[turnstile]]s for these exits are at platform level; passengers wishing to use the express platforms must use a passageway beneath the platforms and tracks. These entrances utilize the northern portions of the platforms. There is also a supplementary and handicapped-accessible entrance to the Penn Station complex in general from 34th Street.<ref name=MTA-PennStation-2015/>
* 33rd Street: There are three direct entrances from the street at 33rd Street and Seventh Avenue. As a replacement for the southwestern corner's lack of an entrance, there is an underground entrance directly connecting the station with the Long Island Rail Road concourse in the Penn Station complex. The turnstiles for this entrance are located below the track level and utilize the central portions of the platforms.<ref name=MTA-PennStation-2015/>
* 32nd Street: The main entrance to the Penn Station complex is located on the western end of 32nd Street. From there, passengers may go through the New Jersey Transit and Long Island Rail Road concourses and use the entrance to this station at the end of the latter's concourse. There is also a smaller exit from the station at the southern ends of the platforms that connects with the end of the New Jersey Transit concourse where it meets the Long Island Rail Road underneath the main corridor in the station that connects New Jersey Transit and Amtrak. There is also an entrance on the north side of 32nd Street between Seventh and Sixth Avenues.<ref name=MTA-PennStation-2015/>

==Ridership==
34th Street–Penn Station on the Broadway–Seventh Avenue Line is continually ranked as one of the busiest stations in the subway system. In 2016, it was the fifth-busiest subway station, with 27,741,367 riders as recorded by the [[Metropolitan Transportation Authority]].<ref name="2016-rider"/> By comparison, its sister station on the Eighth Avenue Line is ranked sixth-busiest, with 25,183,869 passengers.<ref name="2016-rider"/> When the Broadway–Seventh Avenue Line station was a shuttle stop before the rest of the South Ferry extension opened, ridership was quite low; in its first year of operation, only 78,121 boardings were recorded.<ref name="ridership2">{{cite web|url=http://transit.frumin.net/subway/growth/nyc-station-ridership.xls.zip |title=1904-2006 ridership figures |publisher=Metropolitan Transportation Authority |accessdate=August 28, 2008 |archiveurl=https://web.archive.org/web/20110723132222/http://transit.frumin.net/subway/growth/nyc-station-ridership.xls.zip |archivedate=July 23, 2011 |url-status=dead }}</ref>

==References==
{{Reflist}}

==External links==
{{commonscat|34th Street – Penn Station (IRT Broadway – Seventh Avenue Line)}}
*{{NYCS ref|http://www.nycsubway.org/perl/stations?6:3134|IRT West Side Line|34th Street/Penn Station}}
*nycsubway.org – [http://www.nycsubway.org/perl/artwork_show?42 When the Animals Speak Artwork by Elizabeth Grajales (1998)]
*nycsubway.org – [http://www.nycsubway.org/perl/artwork_show?188 A Bird's Life Artwork by Elizabeth Grajales (1997)]
*Station Reporter – [https://web.archive.org/web/20060924173239/http://www.stationreporter.net/1train.htm 1 Train]
*[https://maps.google.com/maps?ie=UTF8&ll=40.751004,-73.990613&spn=0.00382,0.013433&z=17&layer=c&cbll=40.751016,-73.99064&panoid=Y2X89P924U4rav9Pw2REtA&cbp=12,87.27,,0,2.24 34th Street entrance from Google Maps Street View]
*[https://maps.google.com/maps?ie=UTF8&ll=40.749947,-73.991225&spn=0.00382,0.013433&z=17&layer=c&cbll=40.750321,-73.991181&panoid=V7zzTJlGaklNrDnjFNfb9A&cbp=12,343.53,,0,1.05 33rd Street entrance from Google Maps Street View]
* [http://www.google.com/maps/@40.7511675,-73.99028,3a,75y,116.83h,91.8t/data=!3m8!1e1!3m6!1s-GDXHSRNR9Bc%2FV42i3zQ5jKI%2FAAAAAAAAKyU%2FQh7PN7yu1s043j9j8uS7pNXe6N4fekVOQCLIB!2e4!3e11!6s%2F%2Flh4.googleusercontent.com%2F-GDXHSRNR9Bc%2FV42i3zQ5jKI%2FAAAAAAAAKyU%2FQh7PN7yu1s043j9j8uS7pNXe6N4fekVOQCLIB%2Fw203-h100-p-k-no%2F!7i9728!8i4864!4m3!8m2!3m1!1e1!6m1!1e1 Platforms from Google Maps Street View]

{{NYCS stations navbox by service|l1=y|l2=y|l3=y}}
{{NYCS stations navbox by line|7ave=yes}}

{{DEFAULTSORT:34th Street-Penn Station (Irt Broadway-Seventh Avenue Line)}}

[[Category:1918 establishments in New York (state)]]
[[Category:IRT Broadway–Seventh Avenue Line stations]]
[[Category:Midtown Manhattan]]
[[Category:New York City Subway stations in Manhattan|34th Street - Penn Station (IRT Broadway–Seventh Avenue Line)]]
[[Category:New York City Subway stations located underground]]
[[Category:Railway stations in the United States opened in 1918]]
[[Category:Seventh Avenue (Manhattan)]]


{{short description|Policy on permits required to enter Northern Cyprus}}
{{Politics of Northern Cyprus |expanded = }}
{{multiple image
| footer    = Entry and exit stamps on an entry-exit form issued to a Swedish identity card holder.
| width     = 200
| image1    = Nordzypriotischer Einreisestempel.png
| alt1      = Entry stamp
| image2    = Nordzypriotischer Ausreisestempel.png
| alt2      = Exit stamp
}}
Most of the visitors to '''[[Northern Cyprus]]''' do not need to obtain a [[visa (document)|visa]] in advance for short visits.

==Visa policy==
Citizens of all countries may enter without a visa for up to 90 days except for the citizens of the following countries:<ref>{{cite web|url=http://mfa.gov.ct.tr/consular-info/visa-regulations/|title=VISA Regulations - Turkish Republic of Northern Cyprus|author=|date=|website=gov.ct.tr}}</ref>

*{{flag|Armenia}}
*{{flag|Nigeria}}
*{{flag|Syria}}<ref>{{Cite web|url=https://mfa.gov.ct.tr/tr/suriyenin-vize-uygulanan-ulkeler-arasina-dahil-edilmesi-hk/|title=Suriye’nin vize uygulanan ülkeler arasına dahil edilmesi hk|date=2019-06-21|website=Kuzey Kıbrıs Türk Cumhuriyeti|language=tr-TR|access-date=2019-06-23}}</ref>


They must obtain a visa at one of the [[List of diplomatic missions of Northern Cyprus|diplomatic mission of Northern Cyprus]].

Nationals of {{flag|Bangladesh}} are refused entry to Northern Cyprus by air (it is unclear whether this also applies to sea and land entries) <ref>[https://www.timaticweb.com/cgi-bin/tim_website_client.cgi?FullText=1&COUNTRY=CY&SECTION=PA&SUBSECTION=RE&user=FLIGHTWORX&subuser=FLIGHTWORX]</ref>

Visa on arrival was issued when crossing from Southern Cyprus until May 2015.<ref>{{cite web|url=http://www.ansamed.info/ansamed/en/news/sections/politics/2015/05/15/cyprus-talks-no-more-paperwork-required-at-crossing-points_683b77b5-9f95-4a23-8e8a-696d7f0ca0b3.html|title=Cyprus talks lead to visa-free transit points - Politics - ANSAMed.it|author=|date=|website=www.ansamed.info}}</ref>

Visitors given less than 90 days on entry but wishing to stay longer (up to 90 days) can apply for extension at the Immigration Department of the Police Headquarters. To stay longer than 90 days, they have to exit and re-enter the country. Overstayers will be issued a visa penalty fine (100.23 Turkish lira for each day) that has to be paid on a future re-entry.

==Passport exemption==

For visits of up to 90 days (i.e. without obtaining a work or residence permit), citizens of
*{{flagicon|Turkey}} [[Turkish nationality law|Turkey]]
*{{flag|United Kingdom}}
*{{flagicon|European Union}} [[Citizenship of the European Union|European Union]] 

may enter the [[TRNC]] using national ID cards instead of a passport.<ref>{{cite web|url=http://mfa.gov.ct.tr/consular-info/visa-regulations/|title=VISA Regulations - Turkish Republic of Northern Cyprus|author=|date=|website=gov.ct.tr}}</ref>

==See also==
{{Wikivoyage|Northern Cyprus}}
* [[Visa requirements for Northern Cypriot citizens]]
* [[Visa policy of Turkey]]
* [[Visa policy of the Schengen Area]]

==References==
{{reflist|30em}}

==External links==
*[https://mfa.gov.ct.tr/consular-info/visa-regulations/ VISA Regulations], Turkish Republic of Northern Cyprus, Ministry of Foreign Affairs
{{Visa policy by country}}
{{Visa Requirements}}

[[Category:Visa policy by country|Northern Cyprus]]
[[Category:Foreign relations of Northern Cyprus]]


{{Cyprus-stub}}


"""
    text = fix_header(fix_quotes(post_clean(simple_clean(source))))

    sections = []
    section = []
    title = ""
    title_2 = ""
    title_3 = ""
    for section_idx, line in enumerate(text.split("\n")):
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

        section_text, section_links = clean(("\n".join(section)).strip())
        title, _ = clean(title)
        doc = nlp(section_text.strip())

        sents = []
        for s in doc.sents:
            if len(sents) > 0:
                if len(str(sents[-1]).strip()) and str(sents[-1]).strip()[-1] != ".":
                    sents[-1] += str(s)
                    continue
            sents.append(str(s))

        all_lines = "\n".join(sents)
        all_lines = re.sub(r'(\\n){3,}', '\n\n', all_lines.strip())
        out_text = ""
        for idx, s in enumerate(all_lines.split("\n")):
            out_text += str(idx) + "\t" + s.strip() + "\n"

        print(out_text)