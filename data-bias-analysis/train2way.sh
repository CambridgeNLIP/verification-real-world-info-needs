allennlp train -f -s 2way_$SEED --include-package modelling configs/claim_only_2way.jsonnet #-o  '{"trainer": {"random_seed": '${SEED}', "numpy_seed": '${SEED}', "pytorch_seed": '${SEED}'}}'
