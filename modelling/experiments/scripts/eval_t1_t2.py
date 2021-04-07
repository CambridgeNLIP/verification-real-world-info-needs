import argparse
import collections, codecs, json
from sklearn.metrics import f1_score, recall_score, classification_report
import os
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--evidence', type=str, required=True)
parser.add_argument('--veracity', type=str, required=True)
parser.add_argument('--variant', type=str, required=True)
args = parser.parse_args()

pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../predictions')
metrics_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../metrics')

ev_stance_dict = {
    'supporting': 'evidence', 'refuting': 'evidence', 'neutral': 'no-evidence'
}

ev_dict_binary = {
    0: 'no-evidence', 1: 'evidence'
}


def load(src, is_evidence, evidence_variant=None):
    with codecs.open(src, encoding='utf-8') as f_in:
        data = [json.loads(line.strip()) for line in f_in.readlines()]

    if is_evidence:
        if evidence_variant == 'binary':
            prediction_converter = ev_dict_binary
        else:
            assert evidence_variant == '3way'
            prediction_converter = ev_stance_dict

    return_dict = dict()
    for sample in data:
        for section in sample['labels']:
            key = (sample['ncid'], section)
            assert key not in return_dict

            if is_evidence:
                return_dict[key] = {
                    'section_label': sample['labels'][section]['section_label'],
                    'sentence_labels': [
                        ev_stance_dict[pred] for pred in sample['labels'][section]['sentence_labels']
                    ],
                    'sentence_predictions': [
                        prediction_converter[pred] for pred in sample['predictions'][section]['sentence_predictions']
                    ]
                }
            else:
                return_dict[key] = {
                    'label': sample['labels'][section],
                    'prediction': sample['verdict_predictions'][section],
                    'evidence': sample['evidence'][section]
                }

    return return_dict


def get_evaluation(veracity_name, evidence_name, veracity, evidence, is_corrected=False):
    # For isolated evaluation
    target_sections = []
    pred_sections = []
    targets_sentence = []
    pred_sentence = []

    # For joint metric
    weighted_tp = collections.defaultdict(list)
    weighted_fp = collections.defaultdict(list)
    weighted_fn = collections.defaultdict(list)
    weighted_scores = []

    assert set(veracity.keys()) == set(evidence.keys())

    for ncid, section in veracity.keys():
        current_veracity = veracity[(ncid, section)]
        current_evidence = evidence[(ncid, section)]

        assert current_veracity['label'] == current_evidence['section_label']

        if not is_corrected:
            if set(current_evidence['sentence_predictions']) == {'no-evidence'}:
                assert len(current_veracity['evidence']) == 0
            else:
                assert len(current_veracity['evidence']) > 0

        # Add information for isolated evaluation
        target_sections.append(current_veracity['label'])
        pred_sections.append(current_veracity['prediction'])
        targets_sentence.extend(current_evidence['sentence_labels'])
        pred_sentence.extend((current_evidence['sentence_predictions']))

        # Accuracy (veracity)
        if current_veracity['label'] == current_veracity['prediction']:
            current_verdict_score = 1.0
        else:
            current_verdict_score = 0.0

        # Evidence score
        # zero division set to 1 because of neutral. But it is a harsh 0/1 criteria for neutrals....
        evidence_f1 = f1_score(y_true=current_evidence['sentence_labels'],
                               y_pred=current_evidence['sentence_predictions'],
                               pos_label='evidence', zero_division=1)

        # Multiply as joint metric
        weighted_verdict_score = current_verdict_score * evidence_f1
        weighted_scores.append(weighted_verdict_score)

        # And for weighted precision / recall / f1
        print('adding label', current_veracity['label'])
        print('adding prediction', current_veracity['prediction'])
        if current_veracity['label'] == current_veracity['prediction']:
            weighted_tp[current_veracity['label']].append(weighted_verdict_score)
        else:
            weighted_fp[current_veracity['prediction']].append(weighted_verdict_score)
            weighted_fn[current_veracity['label']].append(weighted_verdict_score)

    # Results
    assert len(veracity) == len(weighted_scores)

    def print_weighted_results(name, weighted_scores, weighted_tp, weighted_fp, weighted_fn):
        output_json = {
            'num': len(weighted_scores),
            'weighted_accuracy': float(np.sum(weighted_scores) / len(weighted_scores))
        }
        output = f'# Weighted by {name} \n\n'
        output += f'Number of (claim, section): f{len(weighted_scores)}\n\n'
        output += f'Weighted Accuracy: {round(np.sum(weighted_scores) / len(weighted_scores), 3)}\n'
        all_p = []
        all_r = []
        all_f1 = []
        for label in 'supported refuted neutral'.split(' '):
            output_json[label] = {}
            weighted_precision = np.sum(weighted_tp[label]) / (len(weighted_tp[label]) + len(weighted_fp[label]))
            weighted_recall = np.sum(weighted_tp[label]) / (len(weighted_tp[label]) + len(weighted_fn[label]))
            weighted_f1 = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall)

            output_json[label]['weighted_precision'] = float(weighted_precision)
            output_json[label]['weighted_recall'] = float(weighted_recall)
            output_json[label]['weighted_f1'] = float(weighted_f1)

            output += f'({label}): Precision: {round(weighted_precision, 3)}; Recall: {round(weighted_recall, 3)};  F1: {round(weighted_recall, 3)}\n'
            all_p.append(weighted_precision)
            all_r.append(weighted_recall)
            all_f1.append(weighted_f1)
        output += f'(Average): Precision: {round(np.average(all_p, weights=None), 3)}; '
        output += f'Recall: {round(np.average(all_r, weights=None), 3)}; '
        output += f'F1: {round(np.average(all_f1, weights=None), 3)}\n'
        output_json['avg-weighted-precision'] = float(np.average(all_p, weights=None))
        output_json['avg-weighted-recall'] = float(np.average(all_r, weights=None))
        output_json['avg-weighted-f1'] = float(np.average(all_f1, weights=None))
        return output + '\n\n', output_json

    final_output = ''
    out_f1, out_json_f1 = print_weighted_results('F1', weighted_scores, weighted_tp, weighted_fp, weighted_fn)
    final_output += out_f1

    final_output_json = {
        'evidence': evidence_name,
        'veracity': veracity_name,
        'corrected': is_corrected,
        'weighted_f1': out_json_f1,
        'metrics-isolated-veracity': classification_report(
            y_true=target_sections, y_pred=pred_sections, digits=3, output_dict=True
        ),
        'metrics-isolated-evidence': classification_report(
            y_true=targets_sentence, y_pred=pred_sentence, digits=3, output_dict=True
        )
    }

    final_output += '# Verdict (isolated)\n'
    final_output += classification_report(y_true=target_sections, y_pred=pred_sections, digits=3)

    final_output += '\n\n# Evidence (isolated)\n'
    final_output += classification_report(y_true=targets_sentence, y_pred=pred_sentence, digits=3)

    return final_output, final_output_json


veracity_predictions = load(os.path.join(pred_dir, args.veracity + '.jsonl'), is_evidence=False)
evidence_prediction = load(os.path.join(
    pred_dir, args.evidence + '.jsonl'), is_evidence=True, evidence_variant=args.variant
)

# Report without model correction
report_normal, report_normal_json = get_evaluation(
    args.veracity, args.evidence, veracity_predictions, evidence_prediction
)
report_normal_name = os.path.join(metrics_dir, args.veracity + '_joint_report.txt')
with codecs.open(report_normal_name, 'w', encoding='utf-8') as f_out:
    f_out.write(report_normal)

# Report with model correction
# Correct evidence
corrected_evidence = dict()
for key in evidence_prediction:
    claim_section_veracity_label = veracity_predictions[key]['prediction']

    uncorrected_evidence = evidence_prediction[key]
    if claim_section_veracity_label == 'neutral':
        corrected_evidence[key] = {
            'section_label': uncorrected_evidence['section_label'],
            'sentence_labels': uncorrected_evidence['sentence_labels'],
            'sentence_predictions': ['no-evidence'] * len(uncorrected_evidence['sentence_labels'])
        }
    else:
        corrected_evidence[key] = uncorrected_evidence

# And evaluate
report_corrected, report_corrected_json = get_evaluation(
    args.veracity, args.evidence, veracity_predictions, corrected_evidence, is_corrected=True
)
report_corrected_name = os.path.join(metrics_dir, args.veracity + '_joint_corrected_report.txt')
with codecs.open(report_corrected_name, 'w', encoding='utf-8') as f_out:
    f_out.write(report_corrected)

# Write out json metrics
json_out_name = os.path.join(metrics_dir, args.veracity + '_joint_report.json')
print('json_out_name', json_out_name)
with codecs.open(json_out_name, 'w', encoding='utf-8') as f_out:
    json.dump({
        'corrected': report_corrected_json,
        'uncorrected': report_normal_json
    }, f_out)