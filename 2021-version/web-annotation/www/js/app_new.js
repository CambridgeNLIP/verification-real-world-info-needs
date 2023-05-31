var app_new = angular.module("annotateapp",['ngRoute', 'ngCookies']);

if(typeof(String.prototype.trim) === "undefined")
{
    String.prototype.trim = function()
    {
        return String(this).replace(/^\s+|\s+$/g, '');
    };
}

app_new.filter('filterHtmlChars', function(){
   return function(html) {
       var filtered = angular.element('<div>').html(html).text();
       return filtered;
   }
});


app_new.directive('shortcut', function() {
  return {
    restrict: 'E',
    replace: true,
    scope: true,
    link:    function postLink(scope, iElement, iAttrs){
      jQuery(document).on('keypress', function(e){
         scope.$apply(scope.keyPressed(e));
       });


      jQuery(document).on('keydown', function(e){
         scope.$apply(scope.keyDown(e));
       });
    }
  };
});

app_new.directive('ngConfirmClick', [
        function(){
            return {
                link: function (scope, element, attr) {
                    var msg = attr.ngConfirmClick || "Are you sure?";
                    var clickAction = attr.confirmedClick;
                    element.bind('click',function (event) {
                        if ( window.confirm(msg) ) {
                            scope.$eval(clickAction)
                        }
                    });
                }
            };
    }]);


app_new.directive("ngFormCommit", [function () {
    return {
        require: "form",
        link: function ($scope, $el, $attr, $form) {
            $form.commit = function () {
                $el[0].submit();
            };
        }
    };
}]);




app_new.directive('modal', function(){
        return {
            template: '<div class="modal" tabindex="-1" role="dialog" aria-labelledby="myLargeModalLabel" aria-hidden="true"><div class="modal-dialog modal-lg"><div class="modal-content" ng-transclude><div class="modal-header"><button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button><h4 class="modal-title" id="myModalLabel">Modal title</h4></div></div></div></div>',
            restrict: 'E',
            transclude: true,
            replace:true,
            scope:{visible:'=', onShown:'&', onHide:'&'},
            link:function postLink(scope, element, attrs){

                $(element).modal({
                    show: false,
                    keyboard: attrs.keyboard,
                    backdrop: attrs.backdrop
                });

                scope.$watch(function(){return scope.visible;}, function(value){

                    if(value == true){
                        $(element).modal('show');
                    }else{
                        $(element).modal('hide');
                    }
                });

                $(element).on('shown.bs.modal', function(){
                  scope.$apply(function(){
                    scope.$parent[attrs.visible] = true;
                  });
                });

                $(element).on('shown.bs.modal', function(){
                  scope.$apply(function(){
                      scope.onShown({});
                  });
                });

                $(element).on('hidden.bs.modal', function(){
                  scope.$apply(function(){
                    scope.$parent[attrs.visible] = false;
                  });
                });

                $(element).on('hidden.bs.modal', function(){
                  scope.$apply(function(){
                      scope.onHide({});
                  });
                });
            }
        };
    }
);


app_new.directive('modalHeader', function(){
    return {
        template:'<div class="modal-header"><button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button><h4 class="modal-title">{{title}}</h4></div>',
        replace:true,
        restrict: 'E',
        scope: {title:'@'}
    };
});

app_new.directive('modalBody', function(){
    return {
        template:'<div class="modal-body" ng-transclude></div>',
        replace:true,
        restrict: 'E',
        transclude: true
    };
});

app_new.directive('modalFooter', function(){
    return {
        template:'<div class="modal-footer" ng-transclude></div>',
        replace:true,
        restrict: 'E',
        transclude: true
    };

});



app_new.run(function($rootScope, $http, $templateCache) {
    $rootScope.working = false;
    $rootScope.done = false;
    $rootScope.saved_count = 0;
    $rootScope.saved_claims = [];

    var scripts = document.getElementsByTagName('script');
    var index = scripts.length - 1;
    var myScript = scripts[index];
    var queryString = myScript.src.replace(/^[^\?]+\??/,'');

    $http.get('/views/annotate.html?'+queryString, { cache: $templateCache });
    $http.get('/views/welcome.html?'+queryString, { cache: $templateCache });


});



app_new.factory("wikipediaService",["$rootScope", function($rootScope) {
    var service = {
        getWiki: function(http,name,callback) {
            http.get("/wiki/"+name).then(function successCallback(response) {
                callback(response.data)
            }, function errorCallback(response) {
                alert("Unable to add page. Page names are case-sensitive. \n"+response.statusText );
            });
        }
    };
    return service
}]);

app_new.config(['$routeProvider', function($routeProvider) {
    var scripts = document.getElementsByTagName('script');
    var index = scripts.length - 1;
    var myScript = scripts[index];
    // myScript now contains our script object
    var queryString = myScript.src.replace(/^[^\?]+\??/,'');

    $routeProvider.
        when('/', {
            templateUrl: '/views/welcome.html?'+queryString,
            controller: 'WelcomeController'
        }).
        when("/annotate/:annotationTarget/:hitId", {
            templateUrl: '/views/annotate_tb.html?'+queryString,
            controller: 'AnnotateController',
            reloadOnSearch:false
        }).
        when("/annotate/:annotationAssignmentId", {
            templateUrl: '/views/annotate_tb.html?'+queryString,
            controller: 'AnnotateController',
            reloadOnSearch:false
        }).
        when("/annotate", {
            templateUrl: '/views/annotate_tb.html?'+queryString,
            controller: 'AnnotateController',
            reloadOnSearch:false
        }).
        otherwise({
            redirectTo: '/'
        });


}]);

app_new.controller("WelcomeController",function($scope, $http, $location, $rootScope) {
    $scope.annotate = function(testingMode) {
        $location.path("/annotate")
    };

    console.log("Assignment " + $rootScope.annotationAssignmentId);
    console.log("HIT ID " + $rootScope.annotationAssignmentId);

    if ($rootScope.annotationAssignmentId !== undefined) {
        $location.path("/annotate/" + $rootScope.annotationAssignmentId)
    }

});

app_new.factory('claimsService', ['$rootScope', function ($rootScope) {
    var service = {

        getClaim: function(http, id, callback){
            http.get("/claim/"+id).then(function successCallback(response) {
                callback(response.data)
            });
        },

        getActive: function(http, callback) {
            http.get("/assign").then(function successCallback(response) {
                callback(response.data.id);
            })
        },

        setActive: function(http, annotationAssignmentId, callback) {
            http.get("/assign/"+annotationAssignmentId).then(function successCallback(response) {
                callback(response.data.id);
            })
        },

        putAnnotations: function(
            http,
            id,
            submit_type,
            evidence,
            variant,
            certain,
            relevant,
            unsure,
            assignmentId,
            workerId,
            submitUrl,
            timer,
            callback,
            error_callback)
        {
            http.post("/annotate/"+id, {
                "id":id,
                "submit_type":submit_type,
                "evidence":evidence,
                "variant":variant,
                "certain": certain,
                "relevant": relevant,
                "unsure": unsure,
                "workerId": workerId,
                "assignmentId": assignmentId,
                "turkSubmitTo": submitUrl,
                "annotation_time": timer.totalSeconds
            }).then(function successCallback(response){
                callback()
            }, function errorCallback(response) {
                error_callback(response.statusText, response.data)
            })
        }
    };

    return service;

}]);

app_new.factory("timerService", ['$rootScope',function($rootScope) {

    var service  = {
        "service": function () {
            this.minutes = 0;
            this.seconds = 0;
            this.totalSeconds = 0;
            this.paused = false;
            this.timer = "0:00";

            this.incrementTime = function() {
                if (this.paused) {
                    return;
                }

                this.seconds++;
                this.totalSeconds++;

                if (this.seconds >= 60) {
                    this.seconds = 0;
                    this.minutes++;
                }

                this.timer = "" + this.minutes + ":" + (this.seconds > 9 ? this.seconds : "0" + this.seconds);

            };

            this.start = function(onUpdate) {
                var t = this;
                this.interval = setInterval(function() {t.incrementTime(); onUpdate()}, 1000);
            };

            this.pause = function() {
                this.paused = true;
            };

            this.unpause = function() {
                this.paused = false;
            }

        }
    };

    return service;

}]);



app_new.controller("AnnotateController",function($scope, $sce, $cookies, $rootScope, $routeParams, $anchorScroll, $http, $location, $route, claimsService, wikipediaService, timerService, $window) {
    timer = new timerService.service();
    timer.start(function() { $scope.timer = timer.timer; $scope.$apply(); });

    var onFocus = function () {
        // do something
        timer.unpause();
    };

    var onBlur = function () {
        // do something else
        timer.pause();
    };

    var win = angular.element($window);

    win.on("focus", onFocus);
    win.on("blur", onBlur);

    $scope.$on("$destroy", function handler() {
        win.off("focus", onFocus);
        win.off("blur", onBlur);
    });

    $scope.showNextModalFn = false;
    $scope.guidelines = false;
    $scope.keyboard=false;
    $scope.claim_text = "";
    $scope.entity = "";
    $scope.section = "";
    $scope.loading = true;
    $scope.lines = [];
    $scope.evidence = [];
    $scope.testingMode = false;
    $scope.selections = {};
    $scope.label=  {verdict: "", relevant: "", unsure: "", certainty: ""};



    $scope.anyEvidenceSelected = false;

    $scope.currentSentence = -1;

    if($cookies.get("run2") === undefined) {
        $scope.guidelines = true;
    }

    $scope.next = function() {
        $scope.label=  {verdict: "", relevant: "", unsure: "", certainty: ""};
        $scope.showNextModal = true;

    };

    $scope.no_submit = false;


    $scope.keyPressed = function(e) {
        switch (e.which) {
            case 49:
            case 50:
            case 51:
                if (!$scope.showNextModal) {
                    return;
                }

                if ($scope.anyEvidenceSelected) {
                    r = $scope.getLabel(parseInt(e.key));
                    $scope.label.verdict = r;
                } else{
                    if(parseInt(e.key) === 1) {
                        $scope.label.relevant = "not relevant"
                    } else if (parseInt(e.key) === 2 ) {
                        $scope.label.relevant = "relevant"
                    }
                }
                break;
            case 52:
            case 53:
            case 54:
                if (!$scope.showNextModal) {
                    return;
                }

                if ($scope.label.verdict === "true" || $scope.label.verdict === "false") {
                    if(parseInt(e.key) === 4) {
                        $scope.label.certainty = "certain"
                    } else if (parseInt(e.key) === 5 ) {
                        $scope.label.certainty = "uncertain"
                    }
                } else if ($scope.label.verdict === "can't tell") {
                    if(parseInt(e.key) === 4) {
                        $scope.label.unsure = "speculative"
                    } else if (parseInt(e.key) === 5 ) {
                        $scope.label.unsure = "contradictory"
                    } else if (parseInt(e.key)===6) {
                        $scope.label.unsure = "unrelated"
                    }
                }
                break;
            case 13:
                e.preventDefault();
                if (!$scope.showNextModal) {
                    $scope.showNextModal = true;
                }
                break;
            case 19:
            case 83:
                if ($scope.showNextModal && e.ctrlKey) {
                    e.preventDefault();
                    if($scope.loading || $scope.no_submit) {
                        return
                    }
                    if(
                        $scope.anyEvidenceSelected && (
                        ($scope.label.verdict === 'can\'t tell' && $scope.label.unsure) ||
                        (($scope.label.verdict === "true" || $scope.label.verdict === "false") && $scope.label.certainty)) ||
                        !$scope.anyEvidenceSelected && $scope.label.relevant)
                    {
                        $scope.save($scope.xform);
                    } else {
                        alert("Please answer all questions")
                    }
                }
                break;

        }
    };

    $scope.keyDown = function(e) {
        if ($scope.showNextModal) {
            return;
        }
        switch(e.which) {
            case 40:
                e.preventDefault();
                $scope.sentenceState(true);
                break;
            case 38:
                e.preventDefault();
                $scope.sentenceState(false);
                break;
            case 37:
                e.preventDefault();
                $scope.annotate(true);
                break;
            case 39:
                e.preventDefault();
                $scope.annotate(false);
                break;
        }
    };

    $scope.setForm = function(form) {
        $scope.xform = form;
    };

    $scope.showGuidelines = function() {
        $anchorScroll("top");
        $scope.guidelines = !$scope.guidelines;
        if ($scope.keyboard && $scope.guidelines) {
            $scope.keyboard = false;
        }
    };

    $scope.showKeyboard = function() {
        $anchorScroll("top");
        $scope.keyboard = !$scope.keyboard;
        if ($scope.keyboard && $scope.guidelines) {
            $scope.guidelines = false;
        }
    };

    $scope.annotate = function(inc) {
        if ($scope.currentSentence<0) {
            return
        }
        if (inc && $scope.selections[$scope.currentSentence]<1) {
            $scope.selections[$scope.currentSentence] += 1;
        } else if (!inc && $scope.selections[$scope.currentSentence]>-1) {
            $scope.selections[$scope.currentSentence]-=1;
        }

        $scope.updateEvidence();
    };

    $scope.sentenceState = function(next) {
        scrolled = false;
        if (next) {
            if ($scope.currentSentence < $scope.lines.length - 1) {
                $scope.currentSentence++;
                scrolled = true
            }
        } else {
            if ($scope.currentSentence > 0) {
                $scope.currentSentence--;
                scrolled = true
            }
        }

        if (scrolled) {
            $anchorScroll("sentence_" + $scope.currentSentence.toString());
            if ($scope.lines[$scope.currentSentence].trim() === '') {
                $scope.sentenceState(next)
            }
        }
    };

    $scope.getLabel = function (key) {
        switch (key) {
            case 1:
                return "true";
            case 2:
                return "false";
            case 3:
                return "can't tell"
        }
    };

    $scope.save = function($xform) {
        $scope.loading = true;
        $cookies.put("run2",true);

        $scope.onSubmit($xform);

    };

    $scope.updateEvidence = function () {
        $scope.anyEvidenceSelected = Object.values($scope.selections).indexOf(1) >-1 || Object.values($scope.selections).indexOf(-1) >- 1
        values = Object.values($scope.selections);
        var xCount = 0;
        for(var i = 0; i<values.length; ++i) {
            if (values[i] > 0 || values[i] < 0) {
                xCount ++;
            }
        }

        if (xCount>3) {
            alert("It looks like lots of evidence has been selected. Are you sure this is the minimum to decide if the claim is true or false?")
        }
    };

    $scope.isEmpty = function (obj) {
      return Object.keys(obj).length === 0;
    };

    $scope.cancel = function (id) {
        index = $scope.evidence.indexOf(id);
        if (index > -1) {
           $scope.evidence.splice(index, 1);
        }

    };

    $scope.xSubmitUrl = $rootScope.submitUrl+ "/mturk/externalSubmit";

    $scope.trustAction = function (actionURL) {
        return $sce.trustAsResourceUrl(actionURL);
    };

    $scope.onSubmit = function($xform) {
        positive = Object.values($scope.selections).reduce((sum,number) => sum+number);
        ok = true;

        if($scope.no_submit) {

            $scope.loading = false;
            $scope.message = "You tried to submit a HIT when there was a condition preventing submission. Please check!";
            return;
        }


        if (($scope.label.verdict === "true" || $scope.label.verdict === "false") && positive == 0) {
            if(!confirm("You selected equal number of lines as True and False for a claim you labelled as " + $scope.label.verdict  + " are you sure you want to submit this?")) {
                ok=false;

            }

        } else if ($scope.label.verdict === "true" && positive < 0) {
            if(!confirm("You highlighted more lines as false for a claim you labelled as true. Are you sure you want to submit this?")) {
                ok = false;

            }
        } else if ($scope.label.verdict === "false" && positive > 0) {
            if(!confirm("You highlighted more lines as true for a claim you labelled as false. Are you sure you want to submit this?")) {
                ok = false;

            }
        }

        if($rootScope.assignmentId === "ASSIGNMENT_ID_NOT_AVAILABLE") {
            alert("You must accept this HIT before working on it");
            return
        }


        if (!ok) {
            $scope.showNextModal = false;
            $scope.loading = false;

        } else {
            $scope.message = "Submitting HIT";
            $scope.final_form_id = $rootScope.assignmentId;
            $scope.final_form_submitType = $scope.verdict;
            $scope.final_form_selections = JSON.stringify($scope.selections);
            $scope.final_form_timer = JSON.stringify(timer);

            $scope.showNextModal = false;
            $scope.loading = true;

            claimsService.putAnnotations(
                $http,
                $scope.id,
                $scope.label.verdict,
                $scope.selections,
                $rootScope.variant,
                $scope.label.certainty,
                $scope.label.relevant,
                $scope.label.unsure,
                $rootScope.annotationAssignmentId,
                $rootScope.workerId,
                $rootScope.submitUrl,
                timer,
                function () {

                    $scope.message = "Annotation successfully recorded.";
                    $scope.err = "";
                    if ($rootScope.assignmentId !== undefined) {
                        $xform.commit();
                    } else {
                        $route.reload();
                    }

                }, function (err, mesg) {

                    $scope.loading = false;
                    $scope.message = "An error occurred while saving the data. Please report this to the HIT requester if it persistently fails.";
                    $scope.error = err;
                    $scope.real_error = mesg;
                }
            );
        }
    };


    $scope.onSkip = function() {
        $route.reload();
    };

    $scope.getLines = function(wiki, start_line, end_line) {
        $scope.loading = true;
        wikipediaService.getWiki($http,wiki.replace(/\//gi,"_"), function(data) {
            $scope.loading = false;

            if (start_line !== null && end_line !== null && start_line <= end_line) {
                $scope.lines = data.slice(start_line, end_line+1);
            } else {
                $scope.lines = data
            }
        });
    };

    $scope.showNextModalFn = function() {
        console.log("Showing modal");
        $scope.open_modal +=1
    };

    $scope.loadClaim = function(claim) {

        $scope.showNextModal = false;
        $scope.guidelines = false;
        $scope.keyboard=false;
        $scope.claim_text = "";
        $scope.entity = "";
        $scope.section = "";
        $scope.loading = true;
        $scope.lines = [];
        $scope.evidence = [];
        $scope.testingMode = false;
        $scope.selections = {};
        $scope.label=  {verdict: "", relevant: "", unsure: "", certainty: ""};
        $scope.anyEvidenceSelected = false;

        $scope.open_modal = 0;

        $scope.id = claim["id"];
        $scope.claim_text = claim["claim_text"];
        $scope.entity = claim["entity"];
        $scope.section = claim["section"];
        $scope.loading = false;
        $scope.evidence = [];
        $scope.start_line = "start_line" in claim ?  claim["start_line"] : null;
        $scope.end_line = "end_line" in claim ?  claim["end_line"] : null;


        $scope.getLines(claim["wiki"], $scope.start_line, $scope.end_line);

        $scope.$watchGroup(["label.verdict", "label.unsure","showNextModal"], function(newval, oldval, scope) {
            new_verdict = newval[0];
            new_unsure = newval[1];

            positive = Object.values($scope.selections).length > 0 ? Object.values($scope.selections).reduce((sum, number) => sum + number) : 0;
            allsel = Object.values($scope.selections).length > 0 ? Object.values($scope.selections).reduce((sum, number) =>  Math.abs(sum) + Math.abs(number)) : 0;

            if (new_verdict === "true" && positive < 0 && allsel === (0 - positive)) {
                $scope.no_submit = true;
            } else if (new_verdict === "false" && positive > 0 && allsel === positive) {
                $scope.no_submit = true;
            } else if (new_verdict === "can't tell" && new_unsure === "unrelated") {
                $scope.no_submit = true;
            } else {
                $scope.no_submit = false;
            }


        }, true);

    };

    if ($routeParams.annotationAssignmentId !== undefined) {
        if($rootScope.assignmentId === "ASSIGNMENT_ID_NOT_AVAILABLE") {
            $scope.testingMode=true;
        }

        claimsService.setActive($http, $routeParams.annotationAssignmentId, function (id) {
            claimsService.getClaim($http, id, $scope.loadClaim);
        });

    } else {
        claimsService.getActive($http, function (id) {
            claimsService.getClaim($http, id, $scope.loadClaim);
        });
    }


});
