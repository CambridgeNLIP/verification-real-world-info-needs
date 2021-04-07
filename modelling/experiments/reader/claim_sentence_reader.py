from torch.utils.data import Dataset
import jsonlines
import random
import collections


label_encodings = {
    'supporting': 0,
    'refuting': 1,
    'neutral': 2
}


def get_label(label_mode, label):
    if label_mode == 'binary':
        return label == 'supporting' or label == 'refuting'
    elif label_mode == '3way':
        return label_encodings[label]
    else:
        raise NotImplementedError(label_mode)


class ClaimSentenceReader(Dataset):
    """
    Reads a dataset into (claim, sentence) pairs
    """

    def __init__(self, data_path, lower=True, labels_mode='binary'):
        """
        :param lower
            If set to true, all text will be lowercased
        :param labels_mode ('binary' or '3way')
            Will either set binary labels for each (claim, sentence) pair (is useful evidence or not),
            or will use the actual stance as label.
        """
        self.samples = []

        # Load data
        for sample in jsonlines.open(data_path):
            claim_id = sample['ncid']
            claim = sample['claim']
            if lower:
                claim = claim.lower()

            # Consider each claim - section pair ...
            for section_name in sample['text']:
                # ... and here each claim sentence pair
                section_text = sample['text'][section_name]
                section_labels = sample['labels'][section_name]

                for i, sentence in enumerate(section_text['sentences']):
                    if lower:
                        sentence = sentence.lower()

                    # Only keep sentences with content
                    if len(sentence.strip()) == 0:
                        assert section_labels['sentence_labels'][i] is None
                    else:
                        assert section_labels['sentence_labels'][i] is not None
                        final_sample = {
                            'claim_id': claim_id,
                            'claim': claim,
                            'section': section_name,
                            'sentence': sentence,
                            'sentence_idx': i,
                            'label': get_label(labels_mode, section_labels['sentence_labels'][i])
                        }

                        self.samples.append(final_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ClaimSentenceReaderResampling(Dataset):
    """
    Reads a dataset into (claim, sentence) pairs
    """

    def __init__(self, data_path, lower=True, labels_mode='binary'):
        """
        :param lower
            If set to true, all text will be lowercased
        :param labels_mode ('binary' or '3way')
            Will either set binary labels for each (claim, sentence) pair (is useful evidence or not),
            or will use the actual stance as label.
        """
        self.samples_dict = collections.defaultdict(list)
        self.samples = None
        self.labels_mode = labels_mode

        # Load data
        for sample in jsonlines.open(data_path):
            claim_id = sample['ncid']
            claim = sample['claim']
            if lower:
                claim = claim.lower()

            # Consider each claim - section pair ...
            for section_name in sample['text']:
                # ... and here each claim sentence pair
                section_text = sample['text'][section_name]
                section_labels = sample['labels'][section_name]

                for i, sentence in enumerate(section_text['sentences']):
                    if lower:
                        sentence = sentence.lower()

                    if len(sentence.strip()) == 0:
                        assert section_labels['sentence_labels'][i] is None
                    else:
                        assert section_labels['sentence_labels'][i] is not None

                        label = get_label(labels_mode, section_labels['sentence_labels'][i])
                        final_sample = {
                            'claim_id': claim_id,
                            'claim': claim,
                            'section': section_name,
                            'sentence': sentence,
                            'sentence_idx': i,
                            'label': label
                        }

                        self.samples_dict[label].append(final_sample)

        self.start_epoch()

    def start_epoch(self):
        samples = []
        if self.labels_mode == '3way':
            neutral_samples = self.samples_dict[label_encodings['neutral']]
            supporting_samples = self.samples_dict[label_encodings['supporting']]
            refuting_samples = self.samples_dict[label_encodings['refuting']]

            # Just for this dataset
            assert len(supporting_samples) < len(neutral_samples)
            assert len(supporting_samples) > len(refuting_samples)

            samples.extend(supporting_samples)
            samples.extend(refuting_samples)

            random.shuffle(neutral_samples)
            samples.extend(neutral_samples[:len(supporting_samples)])

        else:
            assert self.labels_mode == 'binary'
            assert len(self.samples_dict) == 2
            label_counts = [(key, len(self.samples_dict[key])) for key in self.samples_dict]
            label_counts = sorted(label_counts, key=lambda x: x[1])
            min_label, min_count = label_counts[0]
            max_lbl = label_counts[1][0]

            samples.extend((self.samples_dict[min_label]))
            samples_max = self.samples_dict[max_lbl]
            random.shuffle(samples_max)
            samples.extend(samples_max[:min_count])

        random.shuffle(samples)
        self.samples = samples
        print('Resampled to', len(self.samples), 'samples.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]