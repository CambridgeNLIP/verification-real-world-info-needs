from typing import List


class EncoderClaimSentence:
    """
    Code adapted from VeriSci (https://github.com/allenai/scifact)
    """
    def __init__(self, tokenizer, device, pad_to_max_length=True, max_len=512):
        self.tokenizer = tokenizer
        self.pad_to_max_length = pad_to_max_length
        self.max_len = max_len
        self.device = device

    def encode(self, claims: List[str], sentences: List[str]):
        encoded_dict = self.tokenizer.batch_encode_plus(
            zip(sentences, claims),
            pad_to_max_length=True,
            return_tensors='pt')
        if encoded_dict['input_ids'].size(1) > self.max_len:
            # Too long for the model. Truncate it
            encoded_dict = self.tokenizer.batch_encode_plus(
                zip(sentences, claims),
                max_length=self.max_len,
                truncation_strategy='only_first',
                pad_to_max_length=self.pad_to_max_length,
                return_tensors='pt')
        encoded_dict = {key: tensor.to(self.device) for key, tensor in encoded_dict.items()}
        return encoded_dict