allennlp train -f -s 3way_$SEED --include-package modelling configs/claim_only_3way.jsonnet #-o  '{"trainer": {"random_seed": '${SEED}', "numpy_seed": '${SEED}', "pytorch_seed": '${SEED}'}}'
