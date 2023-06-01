# AmbiFC
This is the repository of our fact-checking dataset [AmbiFC](https://arxiv.org/abs/2104.00640) and includes data and code to reproduce our experimental 
results. To access the code and data of our previous version please go to [the 2021 Version](2021-version).

*This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.*

## The Dataset AmbiFC
You can [download our dataset here](https://drive.google.com/drive/folders/1j1DJlE0lpTdpCvnTHgk1xr4NnbxGp4Wq), which contains all files in the format of `<split>.<subset>.jsonl` 
with the splits *train*, *dev* and *test* and the subsets *certain* and *uncertain*.

These splits include the entire data, including instances marked as relevant with less than five annotations, that 
are excluded from our experiments.

### Getting the right subset of the data
To filter out all relevant samples for a specified subset, download the data into a directory
(here `./data`) and run the following code:
```python
from ambifc.modeling.dataset.samples import get_samples_for_ambifc_subset
from typing import Dict, List


samples: List[Dict] = get_samples_for_ambifc_subset(
    ambifc_subset='ambifc',  # or 'ambifc-certain', 'ambifc-uncertain-5+'
    split='train',  # or 'dev', 'test',
    data_directory='./data'
)
```

Values for the `ambifc_subset` include:

| **Value**               | **Description**                                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------------|
| `'amnbifc-certain'`     | All samples belonging to AmbiFC (certain).                                                                                             |
| `'amnbifc-certain-5+'`  | All samples belonging to AmbiFC (certain) with at least five annotations.                                                              |
| `'amnbifc-uncertain-5+'` | All samples belonging to AmbiFC (uncertain( with at least five annotations.                                                            |
| `'ambifc'`              | The union of AmbiFC certain (all annotations) and AmbiFC uncertain (5+ annotations). This is the full subset used for our experiments. |

### How does the data look like?
The dataset is provided in a .jsonl format. Each line represents an instance consisting of a claim and an annotated 
Wikipedia passage. The same claim is annotated against multiple passages and hence re-occurs multiple times.

An example for how AmbiFC instances looks like is given below:
```json
{
  "claim": "<The claim to verify>",
  "claim_id": "<Numerical ID of the claim>",
  "wiki_page": "<ID to the Wikipedia page of the used evidence (e.g.: 'newwiki.jsonl:5166')>",
  "wiki_section": "<ID to the section of the used evidence (e.g.: newwiki.jsonl:5166:0'')>",
  "wiki_passage": "<ID to the passage of the used evidence (e.g.: 'newwiki.jsonl:5166:0:0')>",
  "entity": "<Entity name of the Wikipedia page>",
  "section": "<Section title of the Wikipedia page>",
  "category": "<Belonging to the uncertain/certain class>",
  "labels": {
    "method": "dawid-skene",
    "passage": "<Passage-level aggregated label (via method specified above) such as 'supporting'>"
  },
  "passage_annotations": [
    {
        "label": "<Passage level individual annotation, e.g. 'neutral'>",
        "worker": "<Annotator ID, e.g. 136>",
        "relevant": "<Indicate if marked as relevant>",
        "certain": "<Indicate if marked as certain/uncertain>",
        "annotation_time": "<Annotation time in seconds>"
    },
    {"more anotations":  "..."}
  ],
  "sentences": {
    "0": "<Sentence at index 0>",
    "1": "<Sentence at index 1>",
    "2": "<Sentence at index 2>",
    "3": "..."
  },
  "sentence_annotations": {
    "0": [
      {
          "annotator": "<Annotator ID, e.g. 136>",
          "annotation": "<Sentence level annotation by specified annotator for sentence 0, e.g. 'neutral'>"
      },
      {"more anotations":  "..."}
    ], 
    "1": ["..."],
    "...": ["..."]
  }
}
```

**NOTE:** Sentences (and sentence annotations) include empty lines that existed in the respective Wikipedia passage:
```json
{
  "sentences": {
    "0": "The Hindu Kush (Pashto and Persian: هندوکش, commonly understood to mean Hindu killers or killer of the Hindus in Persian; ) is an 800-kilometre-long (500 mi) mountain range that stretches through Afghanistan, from its centre to northern Pakistan and into Tajikistan.",
    "1": "",
    "2": "",
    "3": "It forms the western section of the Hindu Kush Himalayan Region (HKH) and is the westernmost extension of the Pamir Mountains, the Karakoram and the Himalayas.",
    "4": "It divides the valley of the Amu Darya (the ancient Oxus) to the north from the Indus River valley to the south.",
    "5": "The range has numerous high snow-capped peaks, with the highest point being Tirich Mir or Terichmir at 7,708 metres (25,289 ft) in the Chitral District of Khyber Pakhtunkhwa, Pakistan.",
    "6": "To the north, near its northeastern end, the Hindu Kush buttresses the Pamir Mountains near the point where the borders of China, Pakistan and Afghanistan meet, after which it runs southwest through Pakistan and into Afghanistan near their border.",
    "7": "The eastern end of the Hindu Kush in the north merges with the Karakoram Range.",
    "8": "Towards its southern end, it connects with the Spin Ghar Range near the Kabul River.",
    "9": "",
    "10": "",
    "11": "The Hindu Kush range region was a historically significant centre of Buddhism with sites such as the Bamiyan Buddhas.",
    "12": "It remained a stronghold of polytheistic faiths until the 19th century.",
    "13": "The range and communities settled in it hosted ancient monasteries, important trade networks, and travellers between Central Asia and South Asia."
  }
}
```
Empty lines are naturally always labeled as *neutral* and ignored from evidence selection or computation of 
the annotator agreement on a sentence level.

## Experiments

### Setup Environment
Recreate the conda environment by running
```shell
conda create --name ambifcEnv --file requirements.txt
```
or install the following libraries and run together with Python 3.9:
- docopt==0.6.2
- transformers==4.27.4
- pytorch==1.11.0
- torchvision==0.12.0
- torchaudio==0.11.0
- cudatoolkit==11.3
- numpy==1.23.5
- datasets==2.11.0
- sentencepiece==0.1.95
- pandas==1.5.2
- scikit-learn==1.2.0
- scipy==1.9.3
- tokenizers==0.11.4
- tqdm==4.65.0
- wandb==0.14.0

### Training

To train a model define the model type as a .json file and train the model with the following command:
````shell
conda activate ambifcEnv
python train-ambifc.py experiment-config/<config>.json
````

The predictions and evaluation will automatically be created. If the model is for veracity prediction of a pipeline 
architecture, the oracle evidence is used for the automatic prediction and evaluation. To make inferences over
automatically selected evidence, this must be run separately.


#### Veracity Prediction
We provide .json files for veracity experiments using the following naming scheme: `veracity-<architecture>-<subset>-<modeling>-s<seed>.json`

| Field | Values |
|---|---|
|architecture| *pipeline*, *full*|
|subset| *ambifc*, *certain*|
|modeling| *single-label*, *multi-label*, *annotation-distillation*|

For temperature scaling, a single-label model must first be trained. Afterward, run:
```shell
python predict_passage_calibration.py temperature-scaling <model-directory> <tuning-data> <test-data> <temperature-min> <temperature-max> <temperature-step> <ambifc_subset>
```

For instance, to run temperature-scaling on the AmbiFC-trained full-text single-label classification model run:
```shell
python predict_passage_calibration.py temperature-scaling \ 
veracity-full-ambifc-single-label-s1 `# Directory name of model predictions` \
predictions.dev.ambifc.veracity_fulltext-ev.jsonl `# Filename of dev-set predictions for tuning t`\
predictions.test.ambifc.veracity_fulltext-ev.jsonl `# Filename on test prediction when applying the found t`\
0.1 5.0 0.1 `# min/max/step of t` \
ambifc `# AmbiFC subset for evaluation`
```

#### Evidence Selection
We provide .json files for evidence selection experiments using the following naming scheme: `evidence-<subset>-<modeling>-s<seed>.json`

|Field | Values                                                      |
|---|-------------------------------------------------------------|
|subset| *ambifc*, *certain*                                         |
|modeling| *stance*, *binary*, *regression*, *annotation-distillation* |

#### Threshold Finding
When training evidence selection via *regression* or *distillation* the threshold *t* must be identified based on the
development set, and applied on the test set for inference. To find *t*, run the following command:

```shell
python select-evidence-threshold.py <config> <subset>
```

For example:
```shell
python select-evidence-threshold.py \
experiment-config/evidence-certain-annotation-distillation-s1.json `# Experiment config file`\
ambifc-certain `# subset used for tuning.`
```


### Inference & Evaluation
To make predictions and evaluate a model run the following command:

```shell
python predict_ambifc.py <config> <split> [--subset=<ambifc-subset>] [--sentences=<sentence-predictions>] [--pred-file-appendix=<appendix>] [--eval] [--overwrite]
```

For example, to create and evaluate the predictions from the pipeline model using annotation distillation
based on the selected evidence using annotation distillation, run:
```shell
python predict_ambifc.py \
experiment-config/veracity-pipeline-ambifc-annotation-distillation-s1.json `# Veracity model configuration`\ 
test `# Make inference on the test split`\
 --subset ambifc `# On the AmbiFC subset`\
 --sentences evidence-ambifc-annotation-distillation-s1/predictions.test.ambifc.evidence.jsonl `# Point to the sentence predictions to use`\
 --pred-file-appendix ev-annotation-distillation-s1 `# An appendix to identify the predictions based on this evidence`\
 --eval `# Do evaluation after training`
```
This will produce the following files in the veracity model's prediction directory:
- predictions.test.ambifc.veracity_ev-annotation-distillation-s1-ev.jsonl (*Predictions*)
- evaluation-predictions.test.ambifc.veracity_ev-annotation-distillation-s1-ev.json (*Evaluation*)


For pipeline models, this requires selected evidence (or use `--sentences oracle` for oracle evidence). For
full-text models use `--sentences fulltext`.

#### Evaluate a single prediction file
To evaluate a single (veracity) prediction file run:
```shell
python evaluate-prediction-file.py eval <path-to-predictions> <path-to-gold-data> <split>
```

For example:
```shell
python evaluate-prediction-file.py eval \
 ../my-subset-predictions.jsonl `# File to be evaluated`\
 ./data `# Directory containing all gold instances`\
 dev `# split in which to find relevant gold instances`
```

### Baselines
To run baselines based on evidence selection models, make sure that the required evidence selection predictions exist.
If you are using a model that requires finding a threshold (*annotation distillation*), run the script 
first, to find the best threshold. Only evidence selection with three-dimensional outputs (*annotation distillation*, 
*stance*) can be used.

To make the baseline predictions run the following command:
```shell
python predict_sentence_threshold_baseline.py <baseline> <prediction-directory> <sentence-prediction-file> <data_directory> <split> <ambifc_subset> [--threshold=<threshold>]
```
whereas `<baseline>` may be one of "majority" or "most-confident". 
For example, to predict the veracity based on the annotation-distillation evidence selection model over all
sentences with a predicted evidence-probability of at least 0.95, run the following:
```shell
python predict_sentence_threshold_baseline.py most-confident \
evidence-certain-annotation-distillation-s1 `# Prediction directory defines in which (sub)directory the baseline predictions should be made` \
./sent_pred/evidence-ambifc-annotation-distillation-s1/predictions.test.ambifc.evidence.jsonl `# Path to sentence predictions`\
./data `# Path to dataset directory`\
test `# Data split of gold data (needed for evaluation)`\
ambifc `# AmbiFC subset (also needed for evaluation)`\
--threshold 0.95 `# Confidence threshold to ignore all sentences with a lower predicted confidence`
```
The predictions and evaluation will be stored in the directories `veracity_baselines` and `veracity_baselines-evaluation`.
