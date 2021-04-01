var app = angular.module("annotateapp",['chart.js','ngRoute','sticky','ngSanitize']);

if(typeof(String.prototype.trim) === "undefined")
{
    String.prototype.trim = function()
    {
        return String(this).replace(/^\s+|\s+$/g, '');
    };
}


app.directive('ngConfirmClick', [
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

app.directive('modal', function(){
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


app.directive('modalHeader', function(){
    return {
        template:'<div class="modal-header"><button type="button" class="close" data-dismiss="modal" aria-label="Close"><span aria-hidden="true">&times;</span></button><h4 class="modal-title">{{title}}</h4></div>',
        replace:true,
        restrict: 'E',
        scope: {title:'@'}
    };
});

app.directive('modalBody', function(){
    return {
        template:'<div class="modal-body" ng-transclude></div>',
        replace:true,
        restrict: 'E',
        transclude: true
    };
});

app.directive('modalFooter', function(){
    return {
        template:'<div class="modal-footer" ng-transclude></div>',
        replace:true,
        restrict: 'E',
        transclude: true
    };

});


app.run(function($rootScope,$http) {
    $rootScope.working = false;
    $rootScope.done = false;

    $rootScope.saved_count = 0;
    $rootScope.saved_claims = [];
});



app.factory("wikipediaService",["$rootScope", function($rootScope) {
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


app.factory('claimMutationAnnotationService', ['$rootScope', function ($rootScope) {

    var service = {
        putClaims: function (http, operator, timer, id,claims,testingMode, callback) {
            if ($rootScope.working) {
                return;
            }
            $rootScope.working = true;
            http.post('/submit-mutations', {"id": id, "operator":operator, "timer":timer, "claims": claims,"testing":testingMode}).then(function successCallback(response) {
                $rootScope.working = false;
                $rootScope.done = true;
                callback(response.data.pos)
            },function errorCallback(response){
                $rootScope.working = false;
                alert("A network error occurred\n" + response.statusText + "\n\nPlease report this and retry submitting claims")
            });
        },

        getClaim: function(http,claim,id,callback) {
            http.get("/mutate/"+claim+"/"+id).then(function(response) {
                callback(response.data);
            });
        }
    };

    return service;

}]);

app.factory('localClaimsService', ['$rootScope', function ($rootScope) {

    var service = {

        model: {
            claims: ""
        },

        SaveState: function () {
            sessionStorage.claims = angular.toJson(service.model);

        },

        RestoreState: function () {
            service.model = angular.fromJson(sessionStorage.claims);
        }
    }

    $rootScope.$on("savestate", service.SaveState);
    $rootScope.$on("restorestate", service.RestoreState);

    return service;
}]);


app.config(['$routeProvider', function($routeProvider) {
    $routeProvider.
        when('/', {
            templateUrl: '/views/welcome.html?n='+Math.random(),
            controller: 'WelcomeController'
        }).
        when("/label-claims/:annotationTarget/:hitId", {
            templateUrl: '/views/annotate-wf2.html?n='+Math.random(),
            controller: 'WF2Controller',
            reloadOnSearch:false
        }).
        when("/label-claims/:annotationAssignmentId", {
            templateUrl: '/views/annotate-wf2.html?n='+Math.random(),
            controller: 'WF2Controller',
            reloadOnSearch:false
        }).
        when("/label-claims-simple/:annotationAssignmentId", {
            templateUrl: '/views/annotate-wf2-simple.html?n='+Math.random(),
            controller: 'WF2Controller',
            reloadOnSearch:false
        }).
        when("/label-claims", {
            templateUrl: '/views/annotate-wf2.html?n='+Math.random(),
            controller: 'WF2Controller',
            reloadOnSearch:false
        }).
        when("/annotate", {
            templateUrl: '/views/annotate.html?n='+Math.random(),
            controller: 'AnnotateController',
            reloadOnSearch:false
        }).
        otherwise({
            redirectTo: '/'
        });


}]);


var generateUUID = function() {

    var d = new Date().getTime();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (d + Math.random()*16)%16 | 0;
        d = Math.floor(d/16);
        return (c=='x' ? r : (r&0x3|0x8)).toString(16);
    });

    return uuid;
};


app.controller("WelcomeController",function($scope,$http,$location, $rootScope) {


    $scope.start2 = function(testingMode) {
        $location.path("/label-claims")
    };

    console.log("Assignment " + $rootScope.annotationAssignmentId);
    console.log("HIT ID " + $rootScope.annotationAssignmentId);

    if ($rootScope.annotationAssignmentId !== undefined) {
        if ($rootScope.variant == 1) {
            $location.path("/label-claims/" + $rootScope.annotationAssignmentId)
        } else if ($rootScope.variant == 2) {
            $location.path("/label-claims-simple/" + $rootScope.annotationAssignmentId)
        }
    }

});


function count_lines(text){
    return (text.match(/^\s*\S/gm) || "").length
}

app.factory('claimsService', ['$rootScope', function ($rootScope) {
    var service = {

        model: {
            claims: []
        },

        SaveState: function () {
            sessionStorage.claims = angular.toJson(service.model);

        },

        RestoreState: function () {
            service.model = angular.fromJson(sessionStorage.claims);


            if(typeof service.model === "undefined") {
                $rootScope.saved_count = 0;
                service.model = {"claims": []}
            }
            $rootScope.saved_count = service.model.claims.length;
        },

        putAnnotation: function(data) {
            service.model.claims.push(data);
            $rootScope.saved_count = service.model.claims.length;
        },

        appendAnnotation: function(data) {
            service.model.claims[service.model.claims.length - 1]["supports"].push(data["supports"]);
            service.model.claims[service.model.claims.length - 1]["refutes"].push(data["refutes"]);
            service.model.claims[service.model.claims.length - 1]["entity"].push(data["entity"]);
            service.model.claims[service.model.claims.length - 1]["sections"].push(data["sections"]);
        },

        reset: function() {
            service.model.claims = []

        },

        getClaim: function(http, id, callback){
            console.log("Get claim");
            http.get("/claim/"+id).then(function successCallback(response) {
                callback(response.data)
            });
        },

        getActive: function(http, callback) {
            http.get("/assign").then(function successCallback(response) {
                console.log(response.data);
                callback(response.data.id);
            })
        },

        setActive: function(http, annotationAssignmentId, callback) {
            console.log("SET ACTIVE")
            http.get("/assign/"+annotationAssignmentId).then(function successCallback(response) {
                console.log(response.data);
                callback(response.data.id);
            })
        },

        putAnnotations: function(
            http,
            id,
            num_sentences_visited,
            num_custom_items_added,
            submit_type,
            selections,
            support_sents,
            refute_sents,
            partial_support_sents,
            partial_refute_sents,
            informative_sents,
            expanded,
            variant,
            timer,
            callback)
        {
            http.post("/annotate/"+id, {
                "id":id,
                "num_sentences_visited":num_sentences_visited,
                "num_custom_items_added":num_custom_items_added,
                "submit_type":submit_type,
                "selections":selections,
                "support_sents":support_sents,
                "refute_sents":refute_sents,
                "partial_support_sents":partial_support_sents,
                "partial_refute_sents":partial_refute_sents,
                "informative_sents":informative_sents,
                "expanded":expanded,
                "variant":variant,
                "annotation_time": timer.totalSeconds

            }).then(function successCallback(response){
                console.log(response);
                callback()
            })

        }
    };

    $rootScope.$on("savestate", service.SaveState);
    $rootScope.$on("restorestate", service.RestoreState);

    return service;
}]);

app.factory("timerService", ['$rootScope',function($rootScope) {

    var service  = {
        "service": function () {
            this.minutes = 0
            this.seconds = 0
            this.totalSeconds = 0

            this.timer = "0:00"

            this.incrementTime = function() {
                this.seconds++;
                this.totalSeconds++;

                if (this.seconds >= 60) {
                    this.seconds = 0;
                    this.minutes++;
                }

                this.timer = "" + this.minutes + ":" + (this.seconds > 9 ? this.seconds : "0" + this.seconds);

            }

            this.start = function(onUpdate) {
                var t = this;
                this.interval = setInterval(function() {t.incrementTime(); onUpdate()}, 1000);
            }

        }
    }

    return service;

}]);


app.directive("ngFormCommit", [function () {
    return {
        require: "form",
        link: function ($scope, $el, $attr, $form) {
            $form.commit = function () {
                $el[0].submit();
            };
        }
    };
}]);


app.controller("AnnotateController",function($scope, $rootScope,$routeParams,$anchorScroll, $http,$location,$route, claimsService, wikipediaService) {
    $scope.claim_text = "Hello world"
    $scope.lines = [{"sentence":"Test"},{"sentence":"test"}]
});

app.controller("WF2Controller",function($scope,$sce, $rootScope,$routeParams,$anchorScroll, $http,$location,$route, claimsService, wikipediaService, timerService) {
    claimsService.RestoreState();
    timer = new timerService.service();
    timer.start(function() { $scope.timer = timer.timer; $scope.$apply() })

    $scope.claim_text = "";
    $scope.entity = "";
    $scope.lines = [];
    $scope.line_links = [];

    if(typeof claimsService.model === "undefined") {
    	claimsService.model = {claims:[]}
    } else {
        $rootScope.saved_claims = claimsService.model.claims.length
    }

    $scope.showModalWiki = false;

    $scope.onAddEvidence = function() {
        $scope.showModalWiki = true;
    };

    $scope.getEntities = function(line) {
        ents = new Set();
        bits = line.split(/\t/);
        if(bits.length>2) {
            for (pos = 3; pos<=bits.length; pos+=2) {
                found = bits[pos];
                ents.add(found);
            }
        }

        return Array.from(ents);
    };


    $scope.emptyDict = true;
    $scope.dictionary = {};
    $scope.combining = false;

    $scope.combinationEvidence = {}

    $scope.partial = function(id) {
        $scope.combining = true;

    };
    $scope.expanded = []
    $scope.getInfo= function(id, golink) {
        $scope.show_partial = false;
        $scope.combining = false;
        $scope.expanded.push(id);
        idx = id;
        id = id.toString();

        if(id===$scope.active) {
            callback = function() { $scope.goto(golink) } || function () { };
            callback();
            return;
        }

        if($scope.active >= 0) {
            if (hasTrueEntries($scope.selections[$scope.active])) {

                r = confirm("The current sentence has unsaved selections from the dictionary.\nThese are linked to the currently selected sentence and could be forgotten or lost if not saved now. \nContinue?")
                if(r) {
                } else {
                    return;
                }
            }

        }


        index = $scope.support_sents.indexOf(id);
        if (index > -1) {
           $scope.support_sents.splice(index, 1);
        }



        index = $scope.refute_sents.indexOf(id);
        if (index > -1) {
           $scope.refute_sents.splice(index, 1);
        }


        if(id>=0) {
            $scope.numberOfSentencesVisited += 1;
        }

        $scope.dictionary = {};
        $scope.originalSelected = [];
        $scope.active = id;
        $scope.active_ref = idx;
        $scope.emptyDict = true;

        if(!($scope.active in $scope.selections)) {
            $scope.selections[$scope.active] = {}
        }

        entities = $scope.getEntities($scope.lines[id]);

        if (entities.length > 0) {
            $scope.loading = true;
            $scope.emptyDict = false;
            $http.get("/dictionary/"+$scope.entity+"/"+$scope.active).then(function successCallback(response){
                $scope.loading = false;
                $scope.dictionary = response.data

                dictKeys = Object.keys($scope.dictionary);

                for(i = 0; i<dictKeys.length; i++) {
                    activeKeys =  Object.keys($scope.selections[$scope.active])

                    if(activeKeys.indexOf(dictKeys[i])==-1) {

                        $scope.selections[$scope.active][dictKeys[i]] ={}
                    }
                }

                if ($scope.active in $scope.customItems) {
                    for (i = 0; i< $scope.customItems[$scope.active].length; i++) {
                        $scope.dictionary[$scope.customItems[$scope.active][i][0]] = $scope.customItems[$scope.active][i][1]
                    }
                }

                callback = function() { $scope.goto(golink) } || function () { }
                callback();

                $scope.emptyDict = (Object.keys($scope.dictionary).length == 0)
            });


        } else {
            if ($scope.active in $scope.customItems) {
                for (i = 0; i< $scope.customItems[$scope.active].length; i++) {
                    $scope.dictionary[$scope.customItems[$scope.active][i][0]] = $scope.customItems[$scope.active][i][1]
                }
            }

            $scope.emptyDict = (Object.keys($scope.dictionary).length == 0)
        }
    };
    $scope.getLinks = function(lines) {
        line_htmls = {};
        for(i=0; i<lines.length; i++) {
            line_entity_alias = {};
            line = lines[i];

            bits = line.split(/\t/);
            if(bits.length>2) {
                for (pos = 3; pos<=bits.length; pos+=2) {
                    found = bits[pos];
                    line_entity_alias[bits[pos-1]] = bits[pos]
                }
            }

            keys = Object.values(line_entity_alias).sort(function(a, b){
                //Longest first
                return b.length - a.length;
            });

            toRemoveK = [];
            for(key1 in keys) {
                for(key2 in keys) {
                    if (keys[key1].length>keys[key2].length) {
                        if(keys[key1].indexOf(keys[key2])!==-1) {
                            toRemoveK.push(keys[key2])
                        }
                    }
                }
            }

            for (item in toRemoveK) {
                keys.splice(keys.indexOf(toRemoveK[item]),1)
            }

            vals = Object.keys(line_entity_alias).sort(function(a, b){
                //Longest first
                return b.length - a.length;
            });

            toRemoveV = [];
            for(key1 in vals) {
                for(key2 in vals) {
                    if (vals[key1].length>vals[key2].length) {
                        if(vals[key1].indexOf(vals[key2])!==-1) {
                            toRemoveV.push(vals[key2])
                        }
                    }
                }
            }

            for (item in toRemoveV) {
                vals.splice(vals.indexOf(toRemoveV[item]),1)
            }

            cuts = [];

            for(j=0; j<vals.length; j++) {
                surface_form = vals[j];
                destination = line_entity_alias[vals[j]];

                if (keys.indexOf(destination)===-1 && toRemoveV.indexOf(surface_form)===-1) {
                    continue;
                }

                start_idx = bits[1].indexOf(surface_form);

                if(start_idx<0) {
                    continue;
                }

                if(start_idx > 2 && bits[1][start_idx-1]=="]" && (bits[1][start_idx-2]=="N" || bits[1][start_idx-2] == "T" || bits[1][start_idx-2]=="R")) {
                    continue;
                }

                before_text = bits[1].substring(0, start_idx)
                link_text = bits[1].substring(start_idx,start_idx+surface_form.length);
                after_text = bits[1].substring(start_idx+surface_form.length)

                cuts.push([start_idx,surface_form.length, "[START]"+destination.split(" ").join("[JOIN]")+"[SEPARATOR]"+link_text.split(" ").join("[JOIN]")+"[END]"]);
            }

            cuts = Object.values(cuts).sort(function(a, b){
                //order by start idx ascending
                return  a[0]-b[0];
            });


            added = 0;
            for (c = 0; c<cuts.length; c++) {
                cut = cuts[c]

                start_idx =  cut[0]
                before_text = bits[1].substring(0, added+start_idx)
                link_text = bits[1].substring(added+start_idx,added+start_idx+cut[1]);
                after_text = bits[1].substring(added+start_idx+cut[1])

                newb1 = before_text+cut[2]+after_text


                added += (newb1.length-bits[1].length)
                bits[1] = newb1
            }

            line_htmls[i] = []
            last_idx = 0;
            while(bits[1].indexOf("[START]") > -1) {
                start_idx = bits[1].indexOf("[START]")
                sep_idx = bits[1].indexOf("[SEPARATOR]")
                end_idx = bits[1].indexOf("[END]")
                last_idx = end_idx;

                if(start_idx>0) {
                    line_htmls[i].push({ "link":null, "text":bits[1].substring(0,start_idx).replace("[JOIN]"," ") })
                }

                if(start_idx>=0) {
                   line_htmls[i].push({ "link": bits[1].substring(start_idx+7,sep_idx).split("[JOIN]").join(" "), "text": bits[1].substring(sep_idx+11,end_idx).split("[JOIN]").join(" ")   })
                }

                if(end_idx>-1) {
                    bits[1] = bits[1].substring(end_idx+5);
                }
            }

            if(bits[1].length>0) {
                line_htmls[i].push({ "link":null, "text":bits[1] })
            }

        }

        return line_htmls;
    };
    $scope.goto = function(entity) {
        $location.hash("dict_"+entity).replace();
        $anchorScroll();

    };

    $scope.isEmpty = function (obj) {
      return Object.keys(obj).length === 0;
    };

    $scope.selections = {};
    $scope.loading = false;
    $scope.support_sents = [];
    $scope.refute_sents = [];
    $scope.partial_support_sents = [];
    $scope.partial_refute_sents = [];
    $scope.partial_sents = [];
    $scope.informative_sents = [];

    $scope.show_partial = false;
    $scope.partial2 = function(id) {
        $scope.show_partial = !$scope.show_partial
        $scope.combining = false;
    }


    $scope.combine = function(id) {
        if ($scope.selections[$scope.active] === undefined) {
            $scope.selections[$scope.active] = {};
        }

        if ($scope.selections[$scope.active][$scope.entity] === undefined) {
            $scope.selections[$scope.active][$scope.entity] = {};
        }
        $scope.selections[$scope.active][$scope.entity][id] = true

        if ($scope.combinationEvidence[id] === undefined) {
            $scope.combinationEvidence[id] = new Set();
        }
        $scope.combinationEvidence[id].add(parseInt($scope.active))
    };

    $scope.uncombine = function(id) {
        if ($scope.selections[$scope.active] === undefined) {
            $scope.selections[$scope.active] = {};
        }

        if ($scope.selections[$scope.active][$scope.entity] === undefined) {
            $scope.selections[$scope.active][$scope.entity] = {};
        }
        $scope.selections[$scope.active][$scope.entity][id] = false;
        $scope.combinationEvidence[id].delete(parseInt($scope.active))
    };

    $scope.clearCombination = function(id) {
        Object.values($scope.combinationEvidence).forEach(function(ev) {
           if (ev.has(id)) {
               ev.delete(id)
           }
        });
        $scope.selections[$scope.active][$scope.entity] = {};
    };

    $scope.supports = function(id) {
        $scope.combining = false;
        index = $scope.refute_sents.indexOf(id);
        if (index > -1) {
           $scope.refute_sents.splice(index, 1);
        }

        index = $scope.partial_support_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_support_sents.splice(index, 1);
        }

        index = $scope.informative_sents.indexOf(id);
        if (index > -1) {
           $scope.informative_sents.splice(index, 1);
        }

        index = $scope.partial_refute_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_refute_sents.splice(index, 1);
        }

        if($scope.support_sents.indexOf(id) === -1) {
            $scope.support_sents.push(id)
        }

        $scope.active = -1;
        $scope.dictionary = {};

    };

    $scope.refutes = function (id) {
        $scope.combining = false;
        index = $scope.support_sents.indexOf(id);
        if (index > -1) {
           $scope.support_sents.splice(index, 1);
        }

        index = $scope.partial_support_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_support_sents.splice(index, 1);
        }

        index = $scope.informative_sents.indexOf(id);
        if (index > -1) {
           $scope.informative_sents.splice(index, 1);
        }

        index = $scope.partial_refute_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_refute_sents.splice(index, 1);
        }

        if($scope.refute_sents.indexOf(id) === -1) {
            $scope.refute_sents.push(id)
        }

        $scope.active = -1;
        $scope.dictionary = {}
    };

    $scope.informative = function (id) {
        $scope.combining = false;
        index = $scope.support_sents.indexOf(id);
        if (index > -1) {
           $scope.support_sents.splice(index, 1);
        }

        index = $scope.partial_support_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_support_sents.splice(index, 1);
        }

        index = $scope.informative_sents.indexOf(id);
        if (index > -1) {
           $scope.informative_sents.splice(index, 1);
        }

        index = $scope.partial_refute_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_refute_sents.splice(index, 1);
        }

        if($scope.informative_sents.indexOf(id) === -1) {
            $scope.informative_sents.push(id)
        }

        $scope.active = -1;
        $scope.dictionary = {}
    };


    $scope.partial_supports = function(id) {
        index = $scope.refute_sents.indexOf(id);
        if (index > -1) {
           $scope.refute_sents.splice(index, 1);
        }

        index = $scope.support_sents.indexOf(id);
        if (index > -1) {
           $scope.support_sents.splice(index, 1);
        }

        index = $scope.informative_sents.indexOf(id);
        if (index > -1) {
           $scope.informative_sents.splice(index, 1);
        }

        index = $scope.partial_refute_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_refute_sents.splice(index, 1);
        }

        if($scope.partial_support_sents.indexOf(id) === -1) {
            $scope.partial_support_sents.push(id)
        }

        $scope.active = -1;
        $scope.dictionary = {};
    };

    $scope.partial_refutes = function (id) {
        index = $scope.support_sents.indexOf(id);
        if (index > -1) {
           $scope.support_sents.splice(index, 1);
        }

        index = $scope.partial_support_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_support_sents.splice(index, 1);
        }

        index = $scope.informative_sents.indexOf(id);
        if (index > -1) {
           $scope.informative_sents.splice(index, 1);
        }

        index = $scope.refute_sents.indexOf(id);
        if (index > -1) {
           $scope.refute_sents.splice(index, 1);
        }

        if($scope.partial_refute_sents.indexOf(id) === -1) {
            $scope.partial_refute_sents.push(id)
        }

        $scope.active = -1;
        $scope.dictionary = {}
    };


    $scope.reduce_count_a= function(a) {
        return a !== undefined && Object.keys(a).length > 0 && Object.keys(a).map(function(key){return a[key]}).some(x=>x);
    };

    $scope.reduce_count=function(a){
        return a!==undefined && Object.keys(a).length >0 && Object.keys(a).map(function(key,idx) {return $scope.reduce_count_a(a[key])}).some(x=>x)
    };

    $scope.cancel = function (id) {
        $scope.combining = false;
        $scope.selections[id] = {};

        Object.values($scope.combinationEvidence).forEach(function(ev) {
           if (ev.has(id)) {
               ev.delete(id)
           }
        });


        index = $scope.support_sents.indexOf(id);
        if (index > -1) {
           $scope.support_sents.splice(index, 1);
        }

        index = $scope.refute_sents.indexOf(id);
        if (index > -1) {
           $scope.refute_sents.splice(index, 1);
        }

        index = $scope.partial_sents.indexOf(id);
        if (index > -1) {
           $scope.partial-sents.splice(index, 1);
        }

        index = $scope.partial_support_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_support_sents.splice(index, 1);
        }

        index = $scope.partial_refute_sents.indexOf(id);
        if (index > -1) {
           $scope.partial_refute_sents.splice(index, 1);
        }

        $scope.active = -1;
        $scope.dictionary = {}
    };

    $scope.saveNei = function() {
        claimsService.putAnnotation({
            "claim_text":$scope.claim_text,
            "supports":[],
            "selections":[],
            "refutes":[]
        });

        claimsService.saveState();
    };

    $scope.saveClaim = function(repeat) {
        if($scope.appending) {
            claimsService.appendAnnotation({
                "supports":$scope.support_sents,
                "entity": $scope.entity,
                "refutes":$scope.refutes_sents,
                "selections": $scope.selections
            });
        } else {
            claimsService.putAnnotation({
                "claim_text": $scope.claim_text,
                "entity": [$scope.entity],
                "supports": [$scope.support_sents],
                "refutes": [$scope.refute_sents],
                "selections": [$scope.selections]
            });
        }

        console.log(claimsService.model);

        claimsService.SaveState();

        $rootScope.saved_claims = claimsService.model.claims.length;
        if (repeat >0) {
            $scope.appending = true;
            $scope.entity = "";
            $scope.lines = [];
            $scope.support_sents = [];
            $scope.refute_sents = [];
            $scope.line_links = [];
        } else {
            $route.reload();
        }
    };

    $scope.xSubmitUrl = $rootScope.submitUrl+ "/mturk/externalSubmit"

    $scope.trustAction = function (actionURL) {
        return $sce.trustAsResourceUrl(actionURL);
    };

    $scope.block_submit = false;
    $scope.onSubmit = function($xform, submit_type) {

        if($rootScope.assignmentId === "ASSIGNMENT_ID_NOT_AVAILABLE") {
            alert("You must accept this HIT before working on it");
            return
        }

        console.log("Xform")
        console.log($scope.xform);
        console.log($xform)

        $scope.final_form_id = $rootScope.assignmentId;
        $scope.final_form_numberOfSentencesVisited = $scope.numberOfSentencesVisited;
        $scope.final_form_numberOfCustomItemsAdded = undefined !== $scope.customItems[$scope.active] ? $scope.customItems[$scope.active].length :0;
        $scope.final_form_submitType = submit_type;
        $scope.final_form_selections = JSON.stringify($scope.selections);
        $scope.final_form_supportSents = JSON.stringify($scope.support_sents);
        $scope.final_form_refuteSents = JSON.stringify($scope.refute_sents);
        $scope.final_form_partialSupportSents = JSON.stringify($scope.partial_support_sents);
        $scope.final_form_partialRefuteSents = JSON.stringify($scope.partial_refute_sents);
        $scope.final_form_informativeSents = JSON.stringify($scope.informative_sents)
        $scope.final_form_timer = JSON.stringify(timer);
        $scope.block_submit = true;

        claimsService.putAnnotations(
                $http,
                $scope.id,
                $scope.numberOfSentencesVisited,
                undefined !== $scope.customItems[$scope.active] ? $scope.customItems[$scope.active].length :0,
                submit_type,
                $scope.selections,
                $scope.support_sents,
                $scope.refute_sents,
                $scope.partial_support_sents,
                $scope.partial_refute_sents,
                $scope.informative_sents,
                $scope.expanded,
                $rootScope.variant,
                timer,
            function() {
                    if ($rootScope.assignmentId !== undefined) {
                        $xform.commit();
                    } else {

                        $route.reload();
                    }

            }
        );

        //claimStatsService.putStats($http,$scope.id.toString(),"wf2",timer.totalSeconds,1, true)

    };


    $scope.download = function() {
        claims = [];
        claimsService.model.claims.forEach(function(v,i) {
            supportEvidenceGroups = [];
            refuteEvidenceGroups = [];

            v["entity"].forEach(function(entity, idx) {
                console.log(v);
                selection = v["selections"][idx];
                supports = v["supports"][idx];
                refutes = v["refutes"][idx];
                supports.forEach(function(entry, idx) {
                    supportGroup = [];
                    supportGroup.push([null, null, entity.replace("(","-LRB-").replace(")","-RRB-"),entry]);
                    keys = Object.keys(selection[entry]);
                    keys.forEach(function(page, idx) {
                        Object.keys(selection[entry][page]).forEach(function(sub_line, idx) {
                            if(selection[entry][page][sub_line]) {
                                supportGroup.push([null, null, page.replace("(","-LRB-").replace(")","-RRB-"), parseInt(sub_line)])
                            }
                        });
                    });

                    supportEvidenceGroups.push(supportGroup)
                });

                refutes.forEach(function(entry, idx) {
                    refuteGroup = [];
                    refuteGroup.push([null, null, entity.replace("(","-LRB-").replace(")","-RRB-"),entry]);
                    keys = Object.keys(selection[entry]);
                    keys.forEach(function(page, idx) {
                        Object.keys(selection[entry][page]).forEach(function(sub_line, idx) {
                            if(selection[entry][page][sub_line]) {
                                refuteGroup.push([null, null, page.replace("(","-LRB-").replace(")","-RRB-"), parseInt(sub_line)])
                            }
                        });
                    });

                    refuteEvidenceGroups.push(refuteGroup);

                });

                if (refutes.length && supports.length ) {

                    claims.push({
                        "claim": v["claim"],
                        "label": "NOT ENOUGH INFO",
                        "evidence": []
                    })

                } else {
                    if (refutes.length) {
                         claims.push({
                            "claim": v["claim_text"],
                            "label": "REFUTES",
                            "evidence": refuteEvidenceGroups
                         })
                    }

                    if (supports.length) {
                        claims.push({
                            "claim": v["claim_text"],
                            "label": "SUPPORTS",
                            "evidence": supportEvidenceGroups
                        })
                    }
                }



            });

        });
        console.log(claims);

        out = "";

        claims.forEach(function(claim) {
           out += angular.toJson(claim);
           out += '\n'
        });

        window.open('data:application/text;charset=utf-8,' + escape(out));

    };

    $scope.clear= function() {
        claimsService.reset();
        claimsService.SaveState();
        $rootScope.saved_claims = 0;
        $route.reload()
    };

    $scope.appending = false;
    $scope.onSkip = function() {
        $route.reload();
    };

    $scope.addCustom = function(url) {
        url = url.replace("https://en.wikipedia.org/wiki/","");
        url = url.replace("http://en.wikipedia.org/wiki/","");
        url = url.replace("en.wikipedia.org/wiki/","");
        url = url.replace("www.wikipedia.org/wiki/","");

        url = url.replace("https://en.m.wikipedia.org/wiki/","");
        url = url.replace("http://en.m.wikipedia.org/wiki/","");
        url = url.replace("en.m.wikipedia.org/wiki/","");
        url = url.replace("www.m.wikipedia.org/wiki/","");

        if (url.trim().length > 0) {
            $scope.addItem(url)
        }
    };

    $scope.addMain = function(url, start_line, end_line) {
        url = url.replace("https://en.wikipedia.org/wiki/","");
        url = url.replace("http://en.wikipedia.org/wiki/","");
        url = url.replace("en.wikipedia.org/wiki/","");
        url = url.replace("www.wikipedia.org/wiki/","");

        url = url.replace("https://en.m.wikipedia.org/wiki/","");
        url = url.replace("http://en.m.wikipedia.org/wiki/","");
        url = url.replace("en.m.wikipedia.org/wiki/","");
        item = url.replace("www.m.wikipedia.org/wiki/","");


        if (item.trim().length > 0) {
            wikipediaService.getWiki($http,item,function (data) {
                if(data == null){
                    alert("Could not find page")
                }

                $scope.entity = data.canonical_entity;
                $scope.entity_display = data.canonical_entity.replace("-LRB-","(").replace("-RRB-",")").replace("_"," ").replace("-COLON-",":");


                if(data.text.trim().length === 0) {
                    data.text = "0\tNo Information"
                }

                clean_text = [];

                lines = data.text.split("\n");
                $scope.start_line = 0
                $scope.end_line = lines.length
                if(start_line !== null && end_line !== null) {
                    $scope.start_line = start_line;
                    $scope.end_line = end_line;
                } else {
                }

                $scope.lines = lines
                $scope.line_links = $scope.getLinks(lines);
                $scope.showModalWiki = false;
                $scope.support_sents = [];
                $scope.refute_sents = [];

            })
        }
    };


    $scope.highlights = new Set();
    $scope.testingMode = false;
    $scope.addOriginal = function() {
        needsVal = false;
        for (i in $scope.support_sents) {
            sid = $scope.support_sents[i];
            if ($scope.entity in $scope.selections[sid]) {
                if ($scope.active in $scope.selections[sid][$scope.entity]) {
                    needsVal = true;
                    break;
                }
            }
        }

        for (i in $scope.refute_sents) {
            sid = $scope.refute_sents[i];

            if ($scope.entity in $scope.selections[sid]) {
                if ($scope.active in $scope.selections[sid][$scope.entity]) {
                    needsVal = true;
                    break;
                }
            }

        }

        if(needsVal) {
            res = confirm("This sentence has already been selected as part of another annotation that uses the original page. Unless you intend to add new information, continuing will result in a duplicate annotation.")
            if(!res) {
                return;
            }
        }
        $scope.addItem($scope.entity);
    };

    $scope.customItems = {};
    $scope.addItem = function(item) {
        if(item in Object.keys($scope.dictionary)) {
            return;
        }

        wikipediaService.getWiki($http,item,function (data) {
            if(data.text.trim().length == 0) {
                data.text = "0\tNo Information"
            }

            clean_text = []
            lines = data.text.split("\n");
            for (i=0;i<lines.length;i++ ) {
                line = lines[i]
                clean_text.push(line.split("\t").length>1? line.split("\t")[1]  :"");
            }
            $scope.dictionary[data.canonical_entity] = clean_text.join("\n");

            if (typeof $scope.customItems[$scope.active] === "undefined") {
                $scope.customItems[$scope.active] = []
            }
            $scope.customItems[$scope.active].push([data.canonical_entity,clean_text.join("\n")])
            $scope.showModal1=false;

        })

    };

    if ($routeParams.annotationAssignmentId !== undefined) {
        console.log($routeParams.annotationTarget)

        if($rootScope.assignmentId === "ASSIGNMENT_ID_NOT_AVAILABLE") {
            $scope.testingMode=true;
        }


        claimsService.setActive($http, $routeParams.annotationAssignmentId, function (id) {
            console.log(id)
            claimsService.getClaim($http, id, function (claim) {
                console.log("Got claim");
                console.log(claim);
                $scope.numberOfSentencesVisited = 0;
                $scope.claim_text = claim["claim_text"];

                start_line = claim["start_line"];
                end_line = claim["end_line"];

                $scope.highlights = claim["highlights"] === null ? new Set() : new Set(claim["highlights"].map(item=>{
                    return item[0] === claim["page"] ? item[1] : -1;
                }));


                $scope.addMain(claim["page"], start_line, end_line);
                $scope.id = claim["id"]

            });
        });

    } else {
        claimsService.getActive($http, function (id) {
            console.log(id)
            claimsService.getClaim($http, id, function (claim) {
                console.log("Got claim");
                console.log(claim);
                $scope.numberOfSentencesVisited = 0;
                $scope.claim_text = claim["claim_text"];

                start_line = claim["start_line"];
                end_line = claim["end_line"];

                $scope.highlights =  claim["highlights"] === null ? new Set() : new Set(claim["highlights"].map(item=>{
                    return item[0] === claim["page"] ? item[1] : -1;
                }));

                $scope.addMain(claim["page"], start_line, end_line);
                $scope.id = claim["id"]


            });
        });

    }


});


function hasTrueEntries(d) {
    keys = Object.keys(d)

    for (i = 0; i<keys.length; i++) {

        o = d[keys[i]]
        okeys = Object.keys(o)
        for (j = 0; j< okeys.length; j++) {
            if(o[okeys[j]]) {
                return true;
            }
        }

    }

    return false;
}


app.filter('paras', function () {
    return function(text){
        text = String(text).trim();

        splits = text.split(/\r?\n/);

        ret = "";
        for(i=0; i<splits.length; i++) {
            ret = ret+"<p id='sentence"+i+">"+splits[i].split('\t')[2]+"</p>"

        }
        return ret;
    }
});
