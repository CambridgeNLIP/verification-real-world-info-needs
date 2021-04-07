# Adapted from VeriSci (https://github.com/allenai/scifact)

"""
Train a binary model to predict whether a sentence can be used as evidence for or against a claim.
neutral evidence sentence are sampled, i.e. the same number of neutral sentences and evidence sentences exist.
"""

import argparse
import random
import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup
import torch
import numpy as np

from helper.claim_sentence_encoder import EncoderClaimSentence
from helper.claim_sentence_eval import EvaluatorClaimSentenceBinary
from helper.trainer import Trainer
from reader.claim_sentence_reader import ClaimSentenceReader, ClaimSentenceReaderResampling


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=True, help='Folder containing the dataset')
parser.add_argument('--dest', type=str, required=True, help='Folder to save the weights')
parser.add_argument('--model', type=str, default='roberta-base')
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--batch-size-gpu', type=int, default=32, help='The batch size to send through GPU')
parser.add_argument('--batch-size-accumulated', type=int, default=32, help='The batch size for each gradient update')
parser.add_argument('--lr-base', type=float, default=2e-5)
parser.add_argument('--lr-linear', type=float, default=2e-5)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Load data
data_train_path = os.path.join(args.data, 'train.jsonl')
data_dev_path = os.path.join(args.data, 'dev.jsonl')

trainset = ClaimSentenceReaderResampling(data_train_path, labels_mode='binary')
devset = ClaimSentenceReader(data_dev_path, labels_mode='binary')

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)

optimizer = torch.optim.Adam([
    {'params': model.roberta.parameters(), 'lr': args.lr_base},  # if using non-roberta model, change the base param path.
    {'params': model.classifier.parameters(), 'lr': args.lr_linear}
])

num_training_steps = int(len(trainset) // args.batch_size_accumulated) * args.epochs
scheduler = get_cosine_schedule_with_warmup(optimizer, 0, num_training_steps)

encoder = EncoderClaimSentence(tokenizer, device)
evaluator = EvaluatorClaimSentenceBinary(encoder, args.batch_size_gpu)

trainer = Trainer(device)
trainer.train(model, tokenizer, trainset, devset,
              args.batch_size_gpu, args.batch_size_accumulated, args.epochs,
              encoder, optimizer, scheduler, evaluator, args.dest, resampling=True)