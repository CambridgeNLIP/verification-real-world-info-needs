# Adapted from VeriSci (https://github.com/allenai/scifact)

from torch.utils.data import Dataset, DataLoader
import jsonlines, random, collections
import numpy as np

label_encodings = {
    'supported': 0,
    'refuted': 1,
    'neutral': 2
}


class ClaimSectionReader(Dataset):
    """
    Reads a dataset into (claim, section) pairs - based on scifact
    """

    def create_non_rationale_sample(self, claim, claim_id, sentences, rationale_indizes, section_name, lower):
        non_rationale_indizes = set(range(len(sentences))) - set(rationale_indizes)
        if len(non_rationale_indizes) > 0:
            k_max = min(self.max_len_wrong_rationale, len(non_rationale_indizes))
            k = random.randint(1, k_max)
            non_rationale_indizes = sorted(random.sample(non_rationale_indizes, k))
            non_rationale = ' '.join([sentences[idx] for idx in non_rationale_indizes])
            if lower:
                non_rationale = non_rationale.lower()

            return {
                'claim': claim,
                'claim_id': claim_id,
                'evidence': non_rationale,
                'label': label_encodings['neutral'],
                'section': section_name
            }

    def __init__(self, data_path, lower=True, max_len_wrong_rationale=2, resample_neutral=True):
        self.samples = []
        self.max_len_wrong_rationale = max_len_wrong_rationale

        rationale_labels = {'supporting', 'refuting'}

        # Load data
        for sample in jsonlines.open(data_path):
            claim_id = sample['ncid']
            claim = sample['claim']
            if lower:
                claim = claim.lower()

            # Consider each claim - section pair ...
            for section_name in sample['text']:
                # In SciFact they add each rationale individually, here we do only have one
                sentence_labels = [lbl for lbl in sample['labels'][section_name]['sentence_labels'] if lbl is not None]
                sentences = [s for s in sample['text'][section_name]['sentences'] if len(s.strip()) > 0]
                assert len(sentence_labels) == len(sentences)

                rationale_indizes = [i for i in range(len(sentence_labels)) if sentence_labels[i] in rationale_labels]
                rationale_sentences = [sentences[i] for i in rationale_indizes]
                claim_section_label = label_encodings[sample['labels'][section_name]['section_label']]

                if len(rationale_sentences) > 0:
                    # Add claim with rationales
                    rationale = ' '.join(rationale_sentences)
                    if lower:
                        rationale = rationale.lower()
                    self.samples.append({
                        'claim': claim,
                        'claim_id': claim_id,
                        'evidence': rationale,
                        'label': claim_section_label,
                        'section': section_name
                    })

                    # Now add neutral rationale for this claim, section
                    if resample_neutral:
                        neutral_sample = self.create_non_rationale_sample(claim, claim_id, sentences,
                                                                          rationale_indizes, section_name, lower)
                        if neutral_sample is not None:
                            self.samples.append(neutral_sample)
                else:
                    # Add random neutral rationale
                    neutral_sample = self.create_non_rationale_sample(claim, claim_id, sentences,
                                                                      rationale_indizes, section_name, lower)
                    if neutral_sample is not None:
                        self.samples.append(neutral_sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]