import logging
from datetime import datetime, timedelta

from bson import ObjectId
from pymongo import MongoClient, ASCENDING, DESCENDING
import os

logger = logging.getLogger("data-service")

class DataService:
    __shared_state = {}

    def __init__(self):
        self.__dict__ = self.__shared_state

        if "client" not in self.__dict__:
            self.__dict__["client"] = MongoClient(
                'mongodb://%s:%s@%s' % (os.getenv("MONGO_USER"), os.getenv("MONGO_PASS"), os.getenv("MONGO_HOST")))

        if "claims" not in self.__dict__:
            self.__dict__["claims"] = self.__dict__["client"][os.getenv("MONGO_DATABASE")].claims

        if "annotations" not in self.__dict__:
            self.__dict__["annotations"] = self.__dict__["client"][os.getenv("MONGO_DATABASE")].annotations

        if "assignments" not in self.__dict__:
            self.__dict__["assignments"] = self.__dict__["client"][os.getenv("MONGO_DATABASE")].assigments

        if "workers" not in self.__dict__:
            self.__dict__["workers"] = self.__dict__["client"][os.getenv("MONGO_DATABASE")].workers

        if "tie_breaker_hits" not in self.__dict__:
            self.__dict__["tie_breaker_hits"] = self.__dict__["client"][os.getenv("MONGO_DATABASE")].tie_breaker_hits

        if "tie_breaker_claims" not in self.__dict__:
            self.__dict__["tie_breaker_claims"] = self.__dict__["client"][os.getenv("MONGO_DATABASE")].tie_breaker_claims

    def get_claim(self,id):
        return self.claims.find_one({"_id":ObjectId(id)})

    def get_assignment(self,username):
        found = self.assignments.find_one({"username": username,"status":"active"})

        if found is not None:
            return found

        claim = self.claims.find_one({"annotation_count":{"$lt":3},"annotator":{"$nin":[username]}}, sort=[("annotation_count", ASCENDING)])
        if claim is None:
            return None

        insert = self.assignments.insert({
            "claim":claim["_id"],
            "created_date": datetime.now(),
            "status":"active",
            "annotator": username
        })

        return self.assignments.find_one({"_id":insert})

    def get_assignment_id(self,assignmentId):
        found = self.assignments.find_one({"_id": ObjectId(assignmentId)})

        if found is not None:
            return found
        return None

    def seed(self):
        self.claims.insert({
            "claim_text":"this is the first test claim",
            "page":"BBC",
            "annotation_count":0
        })


        self.claims.insert({
            "claim_text":"this is the second test claim",
            "page":"Google",
            "annotation_count":0
        })


        self.claims.insert({
            "claim_text":"this is the third test claim",
            "page":"Boris_Johnson",
            "annotation_count":0
        })


        self.claims.insert({
            "claim_text":"this is the fourth test claim",
            "page":"Tony_Blair",
            "annotation_count":0
        })



    def annotate(self, claim, username, content):
        anno_id = self.annotations.insert({
            "claim":ObjectId(claim),
            "variant": content["variant"] if 'variant' in content else None,
            "certain": content["certain"] if 'certain' in content else None,
            "relevant": content["relevant"] if 'relevant' in content else None,
            "unsure": content["unsure"] if 'unsure' in content else None,
            "evidence": content["evidence"],
            "submit_type": content["submit_type"],
            "username": username,
            "annotation_time":content["annotation_time"],
            "worker_id": content["workerId"] if "workerId" in content else None,
            "assignment_id":content["assignmentId"] if "assignmentId" in content else None,
            "created":datetime.now()
        })
        self.claims.update_one({"_id":ObjectId(claim)},
                               {"$inc":{"annotation_count":1},
                                "$push":{"annotator":username,"annotations":anno_id}
                                })
        self.assignments.update({"claim":ObjectId(claim),"annotator":username},{"$set":{"status":"complete", "annotation":anno_id}})
        return anno_id


    def create_assignment(self, sessionId, annotationTarget, hitId):

        claim = self.claims.find_one({"_id":ObjectId(annotationTarget)})

        if claim is None:
            return None

        insert = self.assignments.insert({
            "claim":claim["_id"],
            "created_date": datetime.now(),
            "status":"active",
            "hitId": hitId,
            "annotator": sessionId
        })

        return self.assignments.find_one({"_id":insert})


    def mturk_create_assignment(self, sessionId, annotationTarget, hitId, workerId, assignmentId,variant):
        claim = self.claims.find_one({"_id":ObjectId(annotationTarget)})

        if claim is None:
            return None

        insert = self.assignments.insert({
            "claim":claim["_id"],
            "created_date": datetime.now(),
            "status":"active",
            "hitId": hitId,
            "sessionId": sessionId,
            "workerId":workerId,
            "assignmentId": assignmentId,
            "variant":variant
        })

        return self.assignments.find_one({"_id":insert})


    def create_or_get_worker(self, workerId):
        worker = self.workers.find_one({"worker_id": workerId})
        if worker is None:
            worker_id = self.workers.insert({
                "created_date": datetime.now(),
                "status": "active",
                "worker_id": workerId,
            })
            worker = self.workers.find_one({"_id": worker_id})
        return worker

    def register_worker_session(self,worker_objid, session):
        self.workers.update({"_id": ObjectId(worker_objid)},
                                {"$addToSet": {"sessions": session}})

    def register_worker_hit(self,worker_objid, hit_id, claim_objid):
        self.workers.update({"_id": ObjectId(worker_objid)},
                                {"$addToSet": {"hits": hit_id, "claims":ObjectId(claim_objid)}})

    def get_worker_score(self, worker_objid):
        worker = self.workers.find_one({"_id": worker_objid})

        score = 0
        if "score_true" in worker:
            score += worker["score_true"]
        if "score_false" in worker:
            score += worker["score_false"]
        if "score_can't tell" in worker:
            score += worker["score_can't tell"]
        if "score_" in worker:
            score += worker["score_"]
        if "score_balance" in worker:
            score += worker["score_balance"]

        return score

    def freeze_worker(self, worker_objid):
        self.workers.update({"_id": ObjectId(worker_objid)},
                            {"$set": {"freeze_date": datetime.now()}})

    def timeout_worker(self, worker_objid):
        self.workers.update({"_id": ObjectId(worker_objid)},
                            {"$set": {"timeout_date": datetime.now()}})

    def act_worker(self, worker_objid):
        self.workers.update({"_id": ObjectId(worker_objid)},
                            {"$set": {"act_date": datetime.now()}})


    def block_worker(self, worker_objid):
        self.workers.update({"_id": ObjectId(worker_objid)},
                            {"$set": {"block_date": datetime.now() + timedelta(days=20)}})


    def suspend_worker(self, worker_objid):
        self.workers.update({"_id": ObjectId(worker_objid)},
                            {"$set": {"block_date": datetime.now() + timedelta(days=5)}})


    def assign_worker_score(self, worker_objid, submit_type):
        self.workers.update({"_id": ObjectId(worker_objid)},
                            {"$inc": {"score_{}".format(submit_type):1}})

    def get_claims_for_annotation(self):
        unapproved_workers = self.workers.find({"act_date": {"$exists": False}})
        claims = []
        for worker in unapproved_workers:
            print("Worker {} needs qualification".format(worker))
            if "claims" in worker:
                claims.extend([( worker["worker_id"], cl) for cl in worker["claims"]])
        return claims

    def get_claims_for_annotation_final(self):
        unapproved_workers = self.workers.find({"freeze_date": {"$exists": True}, "act_date": {"$exists": False}})
        claims = []
        for worker in unapproved_workers:
            print("Worker {} needs qualification".format(worker))
            if "claims" in worker:
                claims.extend([( worker["worker_id"], cl) for cl in worker["claims"]])
        return claims

    def unfreeze_worker(self, worker_id):
        self.workers.update({"_id": ObjectId(worker_id)},
                            {"$set": {"unfreeze_date": datetime.now()}})


    def untimeout_worker(self, worker_id):
        self.workers.update({"_id": ObjectId(worker_id)},
                            {"$unset": {"timeout_date":""}})

    def complete_annotation(self, anno_id,keep,status,correction_type, message,response,success):
        self.annotations.update({"_id": ObjectId(anno_id)},
                                {"$set": {"keep": keep,
                                          "status": status,
                                          "correction_type": correction_type,
                                          "message":message,
                                          "response":response,
                                          "upload_success":success},
                                 "$unset": {"auto_accept":''}})

    def get_annotations_from_hit(self, claim, hit):
        annotations = self.annotations.find({"claim":ObjectId(claim["_id"])})
        ret = []
        for annotation in annotations:
            if "assignment_id" in annotation and annotation["assignment_id"] is not None:
                assignment = self.assignments.find_one({"_id":ObjectId(annotation["assignment_id"])})
                if assignment["hitId"] == hit:
                    ret.append(annotation)

        return ret

    def get_annotations_exclude_hits(self, claim, hits):
        annotations = self.annotations.find({"claim":ObjectId(claim["_id"])})
        ret = []
        for annotation in annotations:
            if "assignment_id" in annotation and annotation["assignment_id"] is not None:
                assignment = self.assignments.find_one({"_id":ObjectId(annotation["assignment_id"])})
                if assignment["hitId"] not in hits:
                    ret.append(annotation)
            else:
                ret.append(annotation)

        return ret

    def require_approval(self, ret):
        self.annotations.update({"_id": ObjectId(ret)},
                                {"$set": {"review_required": True}})

    def getAssignmentFromMTurkID(self, mturk):
        return self.assignments.find_one({"assignmentId":mturk})["_id"]

    def get_annotations_from_claim(self, claim):
        annotations = self.annotations.find({"claim": ObjectId(claim["_id"])})
        return list(annotations)

    def does_hitid_exist(self, hitId):
        return self.claims.find({"active_hits": hitId}).count()

    def get_num_submitted_last_hour(self, worker):

        count = self.annotations.find({"worker_id": worker,"created": {"$gt": datetime.now() + timedelta(hours=-1) } }).count()
        logger.info("worker submitted {} in last hour".format(count))
        return count


    def find_timeout_worker(self):
        return self.workers.find({"timeout_date": {"$lt": datetime.now() + timedelta(hours=-1) }})

    def does_tiebreaker_hitid_exist(self, hitId):
        return self.tie_breaker_hits.find({"hit": hitId}).count()

    def transaction_select_one_for(self, workerId):

        return self.tie_breaker_claims.find_one_and_update({"workers": {"$ne": workerId},
                                                         "in_transaction": {"$exists": False},
                                                         "remaining": {"$gt": 0}
                                                         },
                                                        {"$set": {"in_transaction": True,
                                                                  "transaction_start": datetime.now()
                                                                  }
                                                         })

    def complete_tiebreaker(self, claim, worker_id):
        self.tie_breaker_claims.update_one({"claim":ObjectId(claim)},
                                           {"$inc": {"remaining": -1},
                                            "$push": {"workers":worker_id},
                                            "$unset": {"in_transaction":"", "transaction_start":""}
                                            })

    def clear_transactions(self):
        timeouts = self.tie_breaker_claims.find({"in_transaction": True,
                                             "transaction_start": {"$lt": datetime.now() + timedelta(hours=-1)}})
        for timeout in timeouts:
            self.tie_breaker_claims.update_one({"_id": ObjectId(timeout["_id"])},
                                               {"$unset": {"in_transaction": "", "transaction_start": ""}})


