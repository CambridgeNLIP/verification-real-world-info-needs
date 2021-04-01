import logging
import os
import uuid

import boto3
import flask
from flask import Flask, jsonify, request, render_template
from flask import send_from_directory

from annotation.data.fever_db import FEVERDocumentDatabase
from annotation.data.redirect_db import RedirectDatabase
from annotation.data_service import DataService
from mturk.create_qual import create_qualification_type
from util.wiki import get_wiki_clean_db

logger = logging.getLogger(__name__)


def preprocess_line(line):
    if len(line.split("\t")) > 1:
        return line.split("\t")[1]
    return ""


def qualify_worker(client, worker_id, qual, notify=True):
    client.associate_qualification_with_worker(QualificationTypeId=qual, WorkerId=worker_id, SendNotification=notify, IntegerValue=1)

def disqualify_worker(client, worker_id, qual, reason):
    client.disassociate_qualification_from_worker(QualificationTypeId=qual,
                                                  WorkerId=worker_id,
                                                  Reason=reason)


def setup_logging():
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"),
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def boot():
    global db, redirects
    setup_logging()
    logging.getLogger('flask_cors').setLevel(logging.INFO)
    logging.getLogger('waitress').setLevel(logging.INFO)

    app = Flask(__name__,instance_relative_config=True, template_folder="../../templates", static_folder="../../static", static_url_path="/static")
    app.secret_key = "test"

    logger.info("installing redirects")
    redirects_db_loc = os.getenv("REDIRECTS_DB","redirects.db")

    build = int(os.getenv("BUILD","0"))
    logger.info("Build number {}".format(build))

    try:
        if not os.path.exists(redirects_db_loc):
            redirects = RedirectDatabase(redirects_db_loc)
            redirects.install(os.getenv("REDIRECTS_PATH", "data/redirects.txt"))
    except:
        pass

    redirects = RedirectDatabase(redirects_db_loc)
    logger.info("load db")
    print(os.getenv("FEVER_DB", "data/fever.db"))
    db = FEVERDocumentDatabase(os.getenv("FEVER_DB", "data/fever.db"))
    logger.info("done")

    # For uptime monitoring / application health check
    @app.route('/ping')
    def ping():
        return jsonify(ping='pong')

    @app.route('/sping')
    def sping():
        return jsonify(ping='pong')

    # Get an item from the dictionary
    @app.route("/dictionary/<path:entity_path>")
    def get_dictionary(entity_path):

        path_split = entity_path.split("/")
        entity, sentence_id = "/".join(path_split[:-1]), path_split[-1]

        lines = get_wiki_clean_db(db, entity, redirects)["text"].split("\n")

        sentence_id = int(sentence_id)

        ret = dict()
        if sentence_id < len(lines):
            entities = lines[sentence_id].split("\t")[3::2]

            for e1 in entities:
                w = get_wiki_clean_db(db, e1, redirects)
                if w is None:
                    continue
                body = w["text"]

                if w["canonical_entity"].strip().lower() != entity.strip().lower():
                    ret[w["canonical_entity"]] = "\n".join([t[1] if len(t) > 1 else ""
                                                            for t in [b.split("\t")
                                                                      for b in body[:len(body) - 1].split("\n")]])

        return jsonify(ret)

    ds = DataService()

    # Get a wikipedia page
    @app.route("/wiki/<name>")
    def get_wiki(name):
        return jsonify([preprocess_line(l) for l in db.get_doc_lines(name.replace("data_","data/").replace("_wiki","/wiki"))])

    # Static routes for html, css and js
    @app.route('/private', defaults={'path': 'index.html'})
    def send_index(path):
        return send_from_directory(os.path.join(os.getcwd(), 'www'), path)

    # Static routes for html, css and js
    @app.route("/css/<path:path>")
    def send_css(path):
        return send_from_directory(os.path.join(os.getcwd(), 'www', 'css'), path)

    # Static routes for html, css and js
    @app.route("/views/<path:path>")
    def send_view(path):
        return send_from_directory(os.path.join(os.getcwd(), 'www', 'views'), path)

    # Static routes for html, css and js
    @app.route("/js/<path:path>")
    def send_js(path):
        return send_from_directory(os.path.join(os.getcwd(), 'www', 'js'), path)

    @app.route("/assign")
    def assign():
        if "_id" not in flask.session:
            flask.session["_id"] = str(uuid.uuid4())

        obj = ds.get_assignment(flask.session["_id"])
        return jsonify({"id":str(obj["claim"])})

    @app.route("/mturk/variant/<int:vari>")
    def mturk(vari):

        #https://workersandbox.mturk.com/projects/3V3HPF1X70X1YTMIMMIZUO0ZWUPJMS/tasks/3NOEP8XAU4279GP9D0EIRJGVAKWPXH?assignment_id=3C2NJ6JBKAH8RZ8SZYLZMNDK02L2NP&auto_accept=true
        #https://annotation-live.jamesthorne.co.uk/mturk/variant/3?annotationTarget=5ebbd354f7b73f3f55c47f7a&assignmentId=3C2NJ6JBKAH8RZ8SZYLZMNDK02L2NP&hitId=3NOEP8XAU4279GP9D0EIRJGVAKWPXH&workerId=A1IQ49K8NR7EGI&turkSubmitTo=https%3A%2F%2Fworkersandbox.mturk.com
        annotationTarget = request.args.get("annotationTarget").strip()
        hitId = request.args.get("hitId")
        assignmentId = request.args.get("assignmentId")
        workerId = request.args.get("workerId")
        turkSubmitTo = request.args.get("turkSubmitTo")
        ignoreHIT = "ignorehit" in request.args
        if turkSubmitTo is not None and not ignoreHIT:
            if not ds.does_hitid_exist(hitId):
                return render_template("error.html",
                                       error="Could not find HIT task with ID: {}".format(annotationTarget))

        logger.info("Annotate {}".format(annotationTarget))
        if "_id" not in flask.session:
            flask.session["_id"] = str(uuid.uuid4())

        if workerId is not None:
            worker = ds.create_or_get_worker(workerId)
            ds.register_worker_session(worker["_id"],flask.session["_id"])
            ds.register_worker_hit(worker["_id"], hitId, annotationTarget)

        obj = ds.mturk_create_assignment(
                                flask.session["_id"],
                                annotationTarget,
                                hitId,
                                workerId,
                                assignmentId,
                                vari
        )

        if obj is not None:

            annotationAssignmentId = obj["_id"]
            return render_template("annotate.html", **{"hitId":hitId,
                                                   "assignmentId":assignmentId,
                                                   "workerId":workerId,
                                                   "turkSubmitTo":turkSubmitTo,
                                                   "annotationAssignmentId": annotationAssignmentId,
                                                   "variant":vari,
                                                   "returnUrl":turkSubmitTo,
                                                       "build": build,
                                                "qual": False}) # str(worker is not None and "freeze_date" not in worker).lower()
        else:
            return render_template("error.html", error="Could not find annotation task with ID: {}".format(annotationTarget))


    @app.route("/mturk/tiebreaker/<int:vari>")
    def tiebreaker(vari):
        hitId = request.args.get("hitId")
        assignmentId = request.args.get("assignmentId")
        workerId = request.args.get("workerId")
        turkSubmitTo = request.args.get("turkSubmitTo")
        ignoreHIT = "ignorehit" in request.args
        if turkSubmitTo is not None and not ignoreHIT:
            if not ds.does_tiebreaker_hitid_exist(hitId):
                return render_template("error.html", error="Could not find HIT task with ID: {}".format(hitId))


        logger.info("Ger tiebreaker for HIT {}".format(hitId or "anon"))
        annotationTarget = ds.transaction_select_one_for(workerId)
        if annotationTarget is None:
            return render_template("error.html", error="There are no available tiebreaker HITs for you at the this time ({})".format(hitId))
        else:
            annotationTarget = annotationTarget["claim"]

        logger.info("Annotate {}".format(annotationTarget))
        if "_id" not in flask.session:
            flask.session["_id"] = str(uuid.uuid4())

        if workerId is not None:
            worker = ds.create_or_get_worker(workerId)
            ds.register_worker_session(worker["_id"],flask.session["_id"])
            ds.register_worker_hit(worker["_id"], hitId, annotationTarget)


        obj = ds.mturk_create_assignment(
                                flask.session["_id"],
                                annotationTarget,
                                hitId,
                                workerId,
                                assignmentId,
                                vari
        )

        if obj is not None:
            annotationAssignmentId = obj["_id"]
            return render_template("annotate.html", **{"hitId":hitId,
                                                   "assignmentId":assignmentId,
                                                   "workerId":workerId,
                                                   "turkSubmitTo":turkSubmitTo,
                                                   "annotationAssignmentId": annotationAssignmentId,
                                                   "variant":vari,
                                                   "returnUrl":turkSubmitTo,
                                                       "build": build,
                                                "qual": False}) # str(worker is not None and "freeze_date" not in worker).lower()
        else:
            return render_template("error.html", error="Could not find annotation task with ID: {}".format(annotationTarget))


    @app.route("/clear_timeouts")
    def clearTimeouts():
        cnt = 0
        errs = 0
        endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
        client = boto3.client(
            'mturk',
            endpoint_url=endpoint_url, )

        qual = create_qualification_type(client, "Wikipedia Evidence Finding: Soft Block [Timeout]")
        for worker in ds.find_timeout_worker():
            try:
                disqualify_worker(client, worker["worker_id"], qual, "1 hour timeout has passed.")
                ds.untimeout_worker(worker["_id"])
            except Exception as e:
                logger.error("Could not remove timeout for worker {}".format(worker["_id"]))
                logger.error(e)

                errs +=1
            cnt+=1

        ds.clear_transactions()

        if errs > 0:
            return jsonify({"errors":errs,"cnt":cnt}),500
        return jsonify({"count":cnt})


    @app.route("/clear_all_timeouts")
    def clearAllTimeouts():
        cnt = 0
        errs = 0
        endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'
        client = boto3.client(
            'mturk',
            endpoint_url=endpoint_url, )

        qual = create_qualification_type(client, "Wikipedia Evidence Finding: Soft Block [Timeout]")

        response = client.list_workers_with_qualification_type(
            QualificationTypeId=qual,
            Status='Granted'
        )


        for worker in ds.workers.find():
            try:
                resp = disqualify_worker(client, worker["worker_id"], qual, "1 hour timeout has passed.")
                ds.untimeout_worker(worker["_id"])
                logger.info("Removed timeout for worker {}".format(worker))
            except Exception as e:

                logger.error("Could not remove timeout for worker {}".format(worker["_id"]))
                logger.error(e)
                print(e)
                logger.error(str(e))
                errs += 1
            cnt += 1
        if errs > 0:
            return jsonify({"status":response, "errors":errs,"count":cnt}),500
        return jsonify({"status":response, "count": cnt})


    @app.route("/assign/<annotationAssignmentId>")
    def assignTarget(annotationAssignmentId):
        if "_id" not in flask.session:
            flask.session["_id"] = str(uuid.uuid4())

        obj = ds.get_assignment_id(annotationAssignmentId)
        #obj = ds.create_assignment(flask.session["_id"], annotationAssignmentId, None)
        if obj is not None:
            return jsonify({"id":str(obj["claim"])})
        else:
            return None

    @app.route("/assign/<annotationTarget>/<hitId>")
    def assignTargetHIT(annotationTarget, hitId):
        if "_id" not in flask.session:
            flask.session["_id"] = str(uuid.uuid4())

        obj = ds.create_assignment(flask.session["_id"], annotationTarget, hitId)
        if obj is not None:
            return jsonify({"id":str(obj["claim"])})
        else:
            return None

    @app.route("/claim/<claim>")
    def get(claim):
        ret = ds.get_claim(claim)
        if ret is not None:
            start_line = None
            end_line = None
            highlights = None

            if "start_line" in ret:
                start_line = ret["start_line"]

            if "end_line" in ret:
                end_line = ret["end_line"]

            if "highlights" in ret:
                highlights = ret["highlights"]

            return jsonify({"claim_text":ret["claim_text"],
                            "id":str(ret["_id"]),
                            "wiki":ret["wiki"],
                            "entity":ret["entity"],
                            "section": ret["section"] if len(ret["section"].strip()) else "Main",
                            "start_line":start_line,
                            "end_line":end_line,
                            "highlights":highlights
                            })
        return jsonify({})



    @app.route("/annotate/<claim>",methods=["POST"])
    def annotate(claim):
        if "_id" not in flask.session:
            flask.session["_id"] = str(uuid.uuid4())

        content = request.json
        if all(map(lambda a: a in content,[
                                            "id",
                                            "submit_type",
                                            "evidence",
                                            "annotation_time"])):

            if "workerId" not in content:
                content["workerId"] = "000000000"
                content["turkSubmitTo"] = "sandbox"

            ret = ds.annotate(claim,flask.session["_id"],content)
            ds.complete_tiebreaker(claim, content["workerId"])

            if "workerId" in content:
                worker_id = content["workerId"]

                worker = ds.create_or_get_worker(worker_id)

                ds.assign_worker_score(worker["_id"], content["submit_type"])

                logger.info(ds.get_num_submitted_last_hour(worker_id))

                if "sandbox" in content["turkSubmitTo"]:
                    logger.info("Sandbox task")
                    endpoint_url = 'https://mturk-requester-sandbox.us-east-1.amazonaws.com'
                else:
                    logger.info("Real task")
                    endpoint_url = 'https://mturk-requester.us-east-1.amazonaws.com'


                if content["turkSubmitTo"] is not None and "unfreeze_date" not in worker:
                    ds.require_approval(ret)

                    logger.info("Worker score {}".format(ds.get_worker_score(worker["_id"])))


                    if ds.get_worker_score(worker["_id"]) > 60:
                        client = boto3.client(
                            'mturk',
                            endpoint_url=endpoint_url, )

                        try:
                            qual = create_qualification_type(client, "Wikipedia Evidence Finding: Soft Block [Awaiting Review]")
                            ds.freeze_worker(worker["_id"])
                            qualify_worker(client, worker_id, qual)
                        except:
                            pass

                if content["turkSubmitTo"] is not None and "unfreeze_date" in worker:

                    if ds.get_num_submitted_last_hour(worker_id) > 120:
                        client = boto3.client(
                            'mturk',
                            endpoint_url=endpoint_url, )
                        qual = create_qualification_type(client, "Wikipedia Evidence Finding: Soft Block [Timeout]")
                        ds.timeout_worker(worker["_id"])
                        qualify_worker(client, worker_id, qual)
                    pass


            return jsonify({})
        return jsonify({})



    #@app.route("/seed")
    #def seed():
    #    ds.seed()

    return app


