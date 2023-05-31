# Models for Reproduction
To get started clone the repository, download data into the`./data` directory and make sure the files are named as `train.jsonl`, `dev.jsonl`, `test.jsonl`.

All models use `roberta-base`. Code is largely adapted from [VeriSci](https://github.com/allenai/scifact).

## Installation
To reproduce results use **Python 3.6** and install the requirements:
```
pip install -r requirements.txt
```

## Training

### Evidence Extraction

**Parameters:**
* `--data <dataset directory>` point to the dataset directory containing `train.jsonl` and `dev.jsonl`.
* `--dest <destination name>` name of the model. The trained model will be saved at `models/<destination name>/best_model/`

To train a *binary* model for Evidence Extraction predicting whether a sentence can be used as evidence or not, run e.g.
```bash
python experiments/scripts/train_claim_sentence_binary.py --data ./data --dest evidence-extraction-binary
```

To train a *3way* model for Evidence Extraction run e.g.
```bash
python experiments/scripts/train_claim_sentence_3way.py --data ./data --dest evidence-extraction-3way
```

To apply sampling, i.e. not use all neutral sentences, use `train_claim_sentence_binary_resample.py` or `train_claim_sentence_3way_resample.py`.

### Veracity Prediction
During training non-evidence sentences are sampled for the label neutral.

**Parameters**
* `--data <dataset directory>` point to the dataset directory containing `train.jsonl` and `dev.jsonl`.
* `--dest <destination name>` name of the model. The trained model will be saved at `models/<destination name>/best_model/`
* `--max-rationale <default=2>` defines the maximum number of sentences used, if no evidence sentences exist.
* `--resample-neutral <default=0>`If set to `1`, neutral samples will be created for all samples, even if they contain evidence sentences. If set to `0` neutral samples will only created for samples without evidence.

To train a model for Veracity Prediction run e.g.
```bash
python experiments/scripts/train_claim_section.py --data ./data --dest veracity-classifier
```
## Prediction
### Evidence Extraction
**Parameters:**
* `--dataset <dataset directory>` points to the dataset directory.
* `--datasplit <datasplit>` points to the `.jsonl` file within that directory.
* `--model <model>` points to the trained model.
* `--variant <variant>` defines the variant of the trained model (`binary`, `3way` or `oracle`)
* `--name <name>` defines the name of the prediction file. It will be stored at `predictions/<name>.jsonl`.

To create the predictions run e.g.
```bash
python experiments/scripts/predict_claim_sentence.py --dataset ./data --datasplit test.jsonl --model ./models/evidence-extraction-binary/best_model --variant binary --name evidence-extraction-binary
```
This will automatically compute the metrics.

#### TF-IDF
To evaluate different TF-IDF thresholds run 
```bash
python experiments/scripts/tfidf_evidence_extraction.py --dataset ./data  --name tfidf
```
And to extract evidence sentences given a specific threshold, run
```bash
python experiments/scripts/tfidf_evidence_extraction.py --dataset ./data  --name tfidf --threshold 0.2
```
### Veracity Prediction
**Parameters:**
* `--dataset <dataset directory>` points to the dataset directory.
* `--datasplit <datasplit>` points to the `.jsonl` file within that directory.
* `--model <model>` points to the trained model.
* `--variant-evidence <variant>` defines the variant of the trained model (`binary` or `3way`)
* `--predictions <evidence predictions>` point to the extracted evidence sentences
* `--name <name>` defines the name of the prediction file. It will be stored at `predictions/<name>.jsonl`.
* `--variant-verdict <verdict variant>` is either `classifier` when a trained model is used, or `majority` for majority voting (can only be used with 3way/oracle Evidence Extraction)

To create the predictions run e.g.
```bash
python experiments/scripts/predict_claim_section.py --dataset ./data --datasplit test.jsonl --predictions predictions/evidence-extraction-binary.jsonl --variant-evidence binary --name veracity-prediction --model ./models/veracity-classifier/best_model --variant-verdict classifier
```

To compute the overall metrics run e.g.
```bash
python experiments/scripts/eval_t1_t2.py --evidence evidence-extraction-binary --veracity veracity-prediction --variant binary
```
It will automatically produce metrics with and without the correction rule.

## Disclaimer
> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.