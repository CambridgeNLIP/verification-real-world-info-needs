# Adapted from VeriSci (https://github.com/allenai/scifact)

import argparse
import random
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
import torch
import numpy as np

from helper.claim_sentence_encoder import EncoderClaimSentence
from helper.claim_sentence_eval import EvaluatorClaimSentence3Way
from helper.trainer import Trainer
from reader.claim_section_reader import ClaimSectionReader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-base')
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--batch-size-gpu', type=int, default=32, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=32, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=2e-5)
parser.add_argument('--lr-linear', type=float, default=2e-5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--max-rationale', type=int, default=2)
parser.add_argument('--resample-neutral', type=int, default=0)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Load data
data_train_path = os.path.join(args.data, 'train.jsonl')
data_dev_path = os.path.join(args.data, 'dev.jsonl')

max_len_wrong_rationale = args.max_rationale
resample_neutral = args.resample_neutral == 1

trainset = ClaimSectionReader(data_train_path,
                          max_len_wrong_rationale=max_len_wrong_rationale, resample_neutral=resample_neutral)
devset = ClaimSectionReader(data_dev_path,
                            max_len_wrong_rationale=max_len_wrong_rationale, resample_neutral=resample_neutral)


tokenizer = AutoTokenizer.from_pretrained(args.model)
config = AutoConfig.from_pretrained(args.model, num_labels=3)
model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config).to(device)
optimizer = torch.optim.Adam([
    {'params': model.roberta.parameters(), 'lr': args.lr_base},  # if using non-roberta model, change the base param path.
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])

num_training_steps = int(len(trainset) // args.batch_size_accumulated) * args.epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_training_steps)

# This one works for claim sections as well
encoder = EncoderClaimSentence(tokenizer, device)
evaluator = EvaluatorClaimSentence3Way(encoder, args.batch_size_gpu, key_evidence='evidence')

trainer = Trainer(device, key_evidence='evidence')
trainer.train(model, tokenizer, trainset, devset,
              args.batch_size_gpu, args.batch_size_accumulated, args.epochs,
              encoder, optimizer, scheduler, evaluator, args.dest)

