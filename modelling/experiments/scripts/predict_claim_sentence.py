# Adapted from VeriSci (https://github.com/allenai/scifact)

import argparse
import collections, codecs, json
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from helper.claim_sentence_encoder import EncoderClaimSentence
from reader.prediction_reader import SectionPredictionReader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--datasplit', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--variant', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
args = parser.parse_args()

allowed_variants = {'binary', '3way', 'oracle'}
if args.variant not in allowed_variants:
    raise NotImplementedError('invalid variant: ' + args.variant)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_encodings_3way = {
    0: 'supporting',
    1: 'refuting',
    2: 'neutral'
}

label_encodings_binary = {
    0: 'no-evidence',
    1: 'evidence'
}

label_3way_to_binary = {
    'supporting': 'evidence',
    'refuting': 'evidence',
    'neutral': 'no-evidence'
}

# Load data
dataset = SectionPredictionReader(os.path.join(args.dataset, args.datasplit))

if args.variant != 'oracle':
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
    encoder = EncoderClaimSentence(tokenizer, device)
else:
    tokenizer = model = None
    encoder = None

pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../predictions')
metrics_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../metrics')


file_name = args.name
output_name_pred = os.path.join(pred_dir, file_name) + '.jsonl'
output_name_metrics = os.path.join(metrics_dir, file_name) + '.json'
output_name_metrics_report = os.path.join(metrics_dir, file_name) + '_report.txt'

# Predict
do_eval = False
results = collections.defaultdict(list)

with torch.no_grad():
    for data in tqdm(dataset):
        ncid = data['ncid']
        claim = data['claim']
        section = data['section']
        sentences = data['text']

        if 'labels' in data:
            do_eval = True
        gold_labels = data['labels']

        # Get sentence predictions
        if args.variant != 'oracle':

            encoded_dict = encoder.encode([claim] * len(sentences), sentences)

            if args.variant == '3way':
                sentence_scores = torch.softmax(model(**encoded_dict)[0], dim=1).detach().cpu().numpy()
                sentence_predictions = sentence_scores.argmax(axis=1)
                sentence_predictions = [label_encodings_3way[pred] for pred in sentence_predictions]
            elif args.variant == 'binary':
                sentence_scores = model(**encoded_dict)[0]
                sentence_predictions = sentence_scores.argmax(dim=1).tolist()
            else:
                raise NotImplementedError(args.variant)

            sentence_scores = sentence_scores.tolist()

        elif args.variant == 'oracle':
            sentence_scores = None
            sentence_predictions = gold_labels['sentence_labels']
        else:
            raise NotImplementedError(args.variant)

        data['predictions'] = {
            'sentence_predictions': sentence_predictions,
            'sentence_scores': sentence_scores
        }

        results[ncid].append(data)

    if do_eval:
        report_output = ''
        print('Evaluate ... ')
        all_targets = []
        all_predictions = []
        for ncid in results:
            for claim_section in results[ncid]:
                assert len(claim_section['labels']['sentence_labels']) == len(claim_section['predictions']['sentence_predictions'])
                targets = claim_section['labels']['sentence_labels']
                all_targets.extend(targets)
                all_predictions.extend(claim_section['predictions']['sentence_predictions'])

        # Compute binary metrics
        label_names_binary = [False, True]
        if args.variant == 'binary':
            binary_predictions = [label_encodings_binary[pred] for pred in all_predictions]
            binary_targets = [label_3way_to_binary[lbl] for lbl in all_targets]
        else:
            binary_predictions = [label_3way_to_binary[lbl] for lbl in all_predictions]
            binary_targets = [label_3way_to_binary[lbl] for lbl in all_targets]

        report_output += '# Binary metrics\n'
        report_output += classification_report(y_true=binary_targets, y_pred=binary_predictions)

        binary_macro_f1 = f1_score(binary_targets, binary_predictions, zero_division=0, average='macro')
        binary_all_f1 = tuple(f1_score(binary_targets, binary_predictions, zero_division=0, average=None, labels=label_names_binary))
        binary_all_p = tuple(precision_score(binary_targets, binary_predictions, zero_division=0, average=None, labels=label_names_binary))
        binary_all_r = tuple(recall_score(binary_targets, binary_predictions, zero_division=0, average=None, labels=label_names_binary))

        binary_metrics = {
            'f1-macro': binary_macro_f1,
            'precision': dict([(label_names_binary[i], binary_all_p[i]) for i in range(len(label_names_binary))]),
            'recall': dict([(label_names_binary[i], binary_all_r[i]) for i in range(len(label_names_binary))]),
            'f1': dict([(label_names_binary[i], binary_all_f1[i]) for i in range(len(label_names_binary))])
        }

        # compute 3way metrics
        if args.variant == '3way':

            report_output += '\n# 3way metrics\n'
            report_output += classification_report(y_true=all_targets, y_pred=all_predictions)

            label_names_3way = ['supporting', 'refuting', 'neutral']
            t3way_macro_f1 = f1_score(all_targets, all_predictions, zero_division=0, average='macro')
            t3way_all_f1 = tuple(f1_score(all_targets, all_predictions, zero_division=0, average=None, labels=label_names_3way))
            t3way_all_p = tuple(precision_score(all_targets, all_predictions, zero_division=0, average=None, labels=label_names_3way))
            t3way_all_r = tuple(recall_score(all_targets, all_predictions, zero_division=0, average=None, labels=label_names_3way))

            metrics_3way = {
                'f1-macro': t3way_macro_f1,
                'precision': dict([(label_names_3way[i], t3way_all_p[i]) for i in range(len(label_names_3way))]),
                'recall': dict([(label_names_3way[i], t3way_all_r[i]) for i in range(len(label_names_3way))]),
                'f1': dict([(label_names_3way[i], t3way_all_f1[i]) for i in range(len(label_names_3way))])
            }
        else:
            metrics_3way = None

        with codecs.open(output_name_metrics, 'w', encoding='utf-8') as f_out:
            json.dump({
                'binary': binary_metrics, '3way': metrics_3way
            }, f_out)

        # Output classification report
        with codecs.open(output_name_metrics_report, 'w', encoding='utf-8') as f_out:
            f_out.write(report_output + '\n')

    print('Write predictions')
    with codecs.open(output_name_pred, 'w', encoding='utf-8') as f_out:
        for ncid in results:
            claim_sections = results[ncid]
            claim_sample = {
                'ncid': ncid,
                'claim': claim_sections[0]['claim'],
                'labels': {},
                'predictions': {}
            }

            for claim_section in claim_sections:
                section_name = claim_section['section']
                labels = claim_section['labels']
                predictions = claim_section['predictions']

                claim_sample['labels'][section_name] = labels
                claim_sample['predictions'][section_name] = predictions

            f_out.write(json.dumps(claim_sample) + '\n')
    print('Done.')


