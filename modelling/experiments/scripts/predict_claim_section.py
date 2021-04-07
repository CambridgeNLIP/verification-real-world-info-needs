import argparse
import collections, codecs, json
from sklearn.metrics import classification_report
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from helper.claim_sentence_encoder import EncoderClaimSentence
from reader.prediction_reader import VerdictSectionPredictionReader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--datasplit', type=str, required=True)
parser.add_argument('--predictions', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--variant-evidence', type=str, required=True)
parser.add_argument('--variant-verdict', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

assert args.variant_evidence in {'binary', '3way'}
assert args.variant_verdict in {'classifier', 'majority'}

if args.variant_verdict == 'majority':
    assert args.variant_evidence == '3way'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_encodings_3way = {
    0: 'supported',
    1: 'refuted',
    2: 'neutral'
}

evidence_majority_to_verdict = {
    'supporting': 'supported',
    'refuting': 'refuted'
}

pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../predictions')
metrics_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../metrics')

file_name = args.name
output_name_pred = os.path.join(pred_dir, file_name) + '.jsonl'
output_name_metrics = os.path.join(metrics_dir, file_name) + '_report.txt'

dataset = VerdictSectionPredictionReader(
    os.path.join(args.dataset, args.datasplit), args.predictions, args.variant_evidence
)

if args.variant_verdict != 'majority':
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    encoder = EncoderClaimSentence(tokenizer, device)
else:
    tokenizer = model = None
    encoder = None

results = dict()
with torch.no_grad():
    for data in tqdm(dataset):
        ncid = data['ncid']
        claim = data['claim']
        evidence = data['evidence']
        gold_labels = data['labels']

        data['verdict_predictions'] = {}
        sections = sorted(list(gold_labels.keys()))

        # predict neutral by default if no evidence is found
        sections_neutral = [section for section in sections if len(evidence[section]) == 0]
        for section in sections_neutral:
            data['verdict_predictions'][section] = 'neutral'

        # predict the remaining
        sections_predict = [section for section in sections if section not in sections_neutral]
        for section in sections_predict:
            if args.variant_verdict == 'majority':
                # No need for a model
                prediction_count = collections.Counter(data['sentence_predictions_3way'][section])
                assert 'neutral' not in prediction_count

                prediction_count = prediction_count.most_common()
                if len(prediction_count) == 1:
                    data['verdict_predictions'][section] = evidence_majority_to_verdict[prediction_count[0][0]]
                else:
                    # only true and false
                    print('prediction_count', prediction_count)
                    assert len(prediction_count) == 2
                    if prediction_count[0][1] == prediction_count[1][1]:
                        # no majority: neutral
                        data['verdict_predictions'][section] = 'neutral'
                    else:
                        data['verdict_predictions'][section] = evidence_majority_to_verdict[prediction_count[0][0]]
            else:
                # make prediction
                data['verdict_scores'] = {}
                evidences = [evidence[section] for section in sections_predict]
                encoded_dict = encoder.encode([claim] * len(evidences), evidences)

                verdict_scores = torch.softmax(model(**encoded_dict)[0], dim=1).detach().cpu().numpy()
                verdict_predictions = verdict_scores.argmax(axis=1)
                verdict_predictions = [label_encodings_3way[pred] for pred in verdict_predictions]

                verdict_scores = verdict_scores.tolist()
                for i in range(len(verdict_predictions)):
                    data['verdict_predictions'][sections_predict[i]] = verdict_predictions[i]
                    data['verdict_scores'][sections_predict[i]] = verdict_scores[i]

        assert ncid not in results
        results[ncid] = data

# Evaluate
all_targets = []
all_predictions = []
for ncid in results:
    sample = results[ncid]
    for section in sample['labels']:
        all_targets.append(sample['labels'][section])
        all_predictions.append(sample['verdict_predictions'][section])

report = classification_report(y_true=all_targets, y_pred=all_predictions)
print(report)

with codecs.open(output_name_metrics, 'w', encoding='utf-8') as f_out:
    f_out.write(report)

# Output predictions
with codecs.open(output_name_pred, 'w', encoding='utf-8') as f_out:
    for ncid in results:
        f_out.write(json.dumps(results[ncid]) + '\n')

print('Done.')




