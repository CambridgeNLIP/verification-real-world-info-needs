import argparse
import collections, codecs, json, os, random
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--threshold', type=float, default=-1.0)
parser.add_argument('--ngram', type=int, default=1)
parser.add_argument('--seed', type=int, default=1)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)


def load_data_sections(src):
    with codecs.open(src, encoding='utf-8') as f_in:
        return [json.loads(line.strip()) for line in f_in.readlines()]


def get_sentences(data):
    section_dict = collections.defaultdict(set)
    for sample in data:
        for section in sample['text']:
            # add each sentence of each section.
            # Some (claim, section) may have more annotated sentences. Use a set to make sure each sentence is only
            # used once.
            for sentence in sample['text'][section]['sentences']:
                if len(sentence.strip()) > 0:
                    section_dict[section].add(sentence.lower())
    return [sent for section in section_dict for sent in section_dict[section]]


def get_cosine_similarity(claim, sentences, vectorizer):
    evidence_vectors = vectorizer.transform(sentences).todense()
    claim_vector = vectorizer.transform([claim]).todense()

    assert len(sentences) > 0
    assert evidence_vectors.shape[0] == len(sentences)
    assert claim_vector.shape[0] == 1

    return np.ravel(cosine_similarity(claim_vector, evidence_vectors)).tolist()


def add_cosine_similarities(data, vectorizer):
    for sample in data:
        sample['cosine_similarities'] = {}
        for section in sample['labels']:
            cosine_similarities = get_cosine_similarity(
                sample['claim'].lower(), [
                    s.lower() for s in sample['text'][section]['sentences'] if len(s.strip()) > 0
                ], vectorizer
            )
            sample['cosine_similarities'][section] = cosine_similarities


to_binary_label = {
    'supporting': 'evidence', 'refuting': 'evidence', 'neutral': 'no-evidence'
}


def predict_with_threshold(samples, threshold):
    for sample in samples:
        sample['predictions'] = {}
        for section in sample['labels']:
            preds = [
                'evidence' if similarity >= threshold else 'no-evidence'
                for similarity in sample['cosine_similarities'][section]
            ]
            assert len(preds) == len([lbl for lbl in sample['labels'][section]['sentence_labels'] if lbl is not None])
            sample['predictions'][section] = {
                'sentence_predictions': preds
            }


def evaluate_sentence_predictions(samples):
    targets = []
    predictions = []

    for sample in samples:
        for section in sample['labels']:
            sentence_labels = [lbl for lbl in sample['labels'][section]['sentence_labels'] if lbl is not None]
            sentence_predictions = sample['predictions'][section]['sentence_predictions']
            assert len(sentence_labels) == len(sentence_predictions)
            targets.extend(sentence_labels)
            predictions.extend(sentence_predictions)

    targets = [to_binary_label[lbl] for lbl in targets]
    report = classification_report(y_true=targets, y_pred=predictions, digits=3, output_dict=True)
    return report


to_num_pred = {
    'evidence': 1, 'no-evidence': 0
}


def write_predictions(data, dest):
    with codecs.open(dest, 'w', encoding='utf-8') as f_out:
        for sample in data:
            sample.pop('text', None)
            # get same predictions as in binary
            for section in sample['labels']:
                preds = sample['predictions'][section]['sentence_predictions']
                preds = [to_num_pred[p] for p in preds]
                sample['predictions'][section]['sentence_predictions'] = preds
            f_out.write(json.dumps(sample) + '\n')


# Train tf-idf vectorizer on train data
sentences_train = get_sentences(load_data_sections(os.path.join(args.dataset, 'train.jsonl')))
print('Loaded', len(sentences_train), 'training evidence sentences.')
tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, args.ngram))
tfidf_vectorizer = tfidf_vectorizer.fit(sentences_train)

if args.threshold != -1.0:
    # Evaluate on test
    data_test = load_data_sections(os.path.join(args.dataset, 'test.jsonl'))
    add_cosine_similarities(data_test, tfidf_vectorizer)
    predict_with_threshold(data_test, args.threshold)
    report = evaluate_sentence_predictions(data_test)
    print(f'Test performance using threshold={args.threshold}: {report["evidence"]["f1-score"]}')
    all_evidence_metrics = '; '.join([
        str(round(report["evidence"][m], 3)) for m in report["evidence"]
    ])
    all_no_evidence_metrics = '; '.join([
        str(round(report["no-evidence"][m], 3)) for m in report["no-evidence"]
    ])
    print(f' - Evidence: {all_evidence_metrics}')
    print(f' - No-Evidence: {all_no_evidence_metrics}')
    print('---\n')

    # Save
    pred_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../predictions')
    pred_name = os.path.join(pred_dir, args.name)
    write_predictions(data_test, pred_name)

else:
    # Evaluate on dev based on multiple tf-idf thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Vectorize dev data
    data_dev = load_data_sections(os.path.join(args.dataset, 'dev.jsonl'))
    add_cosine_similarities(data_dev, tfidf_vectorizer)

    scores = []
    for t in thresholds:
        predict_with_threshold(data_dev, t)
        report = evaluate_sentence_predictions(data_dev)
        others = ''

        print(f'Dev performance using threshold={t}: {report["evidence"]["f1-score"]}')
        all_evidence_metrics = '; '.join([
            str(round(report["evidence"][m], 3)) for m in report["evidence"]
        ])
        all_no_evidence_metrics = '; '.join([
            str(round(report["no-evidence"][m], 3)) for m in report["no-evidence"]
        ])
        print(f' - Evidence: {all_evidence_metrics}')
        print(f' - No-Evidence: {all_no_evidence_metrics}')
        print('---\n')



