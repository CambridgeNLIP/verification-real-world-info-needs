from torch.utils.data import Dataset
import jsonlines


class SectionPredictionReader(Dataset):
    """
    Used for sentence prediction. Probably change naming.
    """
    def __init__(self, data_path, lower=True):

        # store on a (claim, section) level
        self.samples = []

        for sample in jsonlines.open(data_path):
            claim_id = sample['ncid']
            claim = sample['claim']
            if lower:
                claim = claim.lower()

            # Consider each claim - section pair ...
            for section_name in sample['text']:
                # ... and here each claim sentence pair
                sentences = sample['text'][section_name]['sentences']
                if lower:
                    sentences = [s.lower() for s in sentences]

                sentences = [s for s in sentences if len(s.strip()) > 0]
                sentence_labels = [lbl for lbl in sample['labels'][section_name]['sentence_labels'] if lbl is not None]
                assert len(sentence_labels) == len(sentences)

                final_sample = {
                    'ncid': claim_id,
                    'claim': claim,
                    'section': section_name,
                    'text': sentences,
                    'labels': {
                        'sentence_labels': sentence_labels,
                        'section_label': sample['labels'][section_name]['section_label']
                    }

                }

                self.samples.append(final_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


sentence_encoding_binary_to_binary = {
    0: False,
    1: True
}


stance_label_to_binary = {
    'supporting': True,
    'refuting': True,
    'neutral': False
}


class VerdictSectionPredictionReader(Dataset):
    """
    Reads the predictions from the evicence extraction step to evaluate the (claim, section) verdict.
    """

    def is_sentence_evidence(self, sentence_predictions):
        if self.variant_evidence_extraction == 'binary':
            return [sentence_encoding_binary_to_binary[p] for p in sentence_predictions]
        else:
            assert self.variant_evidence_extraction == '3way'
            return [stance_label_to_binary[p] for p in sentence_predictions]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __init__(self, data_path_full, data_path_predictions,
                 variant_evidence_extraction, lower=True):
        """
        :param data_path_full
            Path to the full dataset
        :param data_path_predictions
            Path to the predictions on a sentence level
        :param variant_evidence_extraction
            Variant of the evidence extraction method ("stance" or "binary"). This will impact how to read the data.
        """
        self.variant_evidence_extraction = variant_evidence_extraction
        assert variant_evidence_extraction in {'3way', 'binary'}

        full_sample_dict = dict()
        prediction_dict = dict()
        self.samples = []

        # Load data
        for sample in jsonlines.open(data_path_full):
            assert sample['ncid'] not in full_sample_dict
            full_sample_dict[sample['ncid']] = sample

        # Load predictions
        for prediction in jsonlines.open(data_path_predictions):
            assert prediction['ncid'] not in prediction_dict
            prediction_dict[prediction['ncid']] = prediction

        assert set(full_sample_dict.keys()) == set(prediction_dict.keys())

        # Create samples
        ncids = sorted(list(prediction_dict.keys()))
        for ncid in ncids:
            prediction = prediction_dict[ncid]
            sample = full_sample_dict[ncid]
            assert set(prediction['predictions'].keys()) == set(sample['labels'].keys())
            assert set(prediction['labels'].keys()) == set(sample['labels'].keys())
            assert set(prediction['predictions'].keys()) == set(sample['text'].keys())

            claim = prediction['claim']
            if lower:
                claim = claim.lower()

            final_sample = {
                'claim': claim,
                'ncid': ncid,
                'evidence': {},
                'labels': {},
            }
            if self.variant_evidence_extraction == '3way':
                final_sample['sentence_predictions_3way'] = {}

            for section in sorted(list(prediction['labels'].keys())):
                assert len(prediction['predictions'][section]['sentence_predictions']) == len([
                    s for s in sample['text'][section]['sentences'] if len(s.strip()) > 0
                ])
                assert len(prediction['predictions'][section]['sentence_predictions']) == len([
                    lbl for lbl in sample['labels'][section]['sentence_labels'] if lbl is not None
                ])
                sentence_is_evidence = self.is_sentence_evidence(
                    prediction['predictions'][section]['sentence_predictions']
                )

                # Get text for evidences
                all_sentences = [s for s in sample['text'][section]['sentences'] if len(s.strip()) > 0]
                assert len(all_sentences) == len(sentence_is_evidence)
                evidence_sentences = [all_sentences[i] for i in range(len(all_sentences)) if sentence_is_evidence[i]]
                evidence = ' '.join(evidence_sentences).strip()
                if lower:
                    evidence = evidence.lower()

                final_sample['evidence'][section] = evidence
                final_sample['labels'][section] = sample['labels'][section]['section_label']

                if self.variant_evidence_extraction == '3way':
                    # additionally give out the stance labels
                    sent_pred = prediction['predictions'][section]['sentence_predictions']
                    assert len(sent_pred) == len(sentence_is_evidence)
                    final_sample['sentence_predictions_3way'][section] = [sent_pred[i] for i in range(len(sent_pred)) if sentence_is_evidence[i]]

            self.samples.append(final_sample)