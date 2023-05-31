# Code adapted from VeriSci (https://github.com/allenai/scifact)

import codecs
from tqdm import tqdm
from torch.utils.data import DataLoader
import os
import json


class Trainer:
    def __init__(self, device, override=0, key_evidence='sentence'):
        self.device = device
        self.model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), './../../models')
        self.override = override
        self.key_evidence = key_evidence

    def train(self, model, tokenizer, trainset, devset, batch_size_gpu, batch_size_accumulated, epochs,
              encoder, optimizer, scheduler, evaluator, dest, resampling=False):

        best_dev_score = -1
        best_dev_metrics = None
        best_epoch = None
        best_num_samples = 0
        save_path = os.path.join(os.path.join(self.model_dir, dest), 'best_model')

        if os.path.exists(save_path) and self.override == 0:
            raise ValueError('A model already exists at ' + save_path)

        metrics = []
        for e in range(epochs):
            batches_count = 0
            if resampling:
                trainset.start_epoch()

            model.train()
            t = tqdm(DataLoader(trainset, batch_size=batch_size_gpu, shuffle=True))
            for i, batch in enumerate(t):
                encoded_dict = encoder.encode(batch['claim'], batch[self.key_evidence])
                loss, logits = model(**encoded_dict, labels=batch['label'].long().to(self.device))
                loss.backward()
                if (i + 1) % (batch_size_accumulated // batch_size_gpu) == 0:
                    batches_count += 1
                    optimizer.step()
                    optimizer.zero_grad()
                    t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')

            scheduler.step()
            train_metrics, train_score = evaluator.evaluate(model, trainset)
            dev_metrics, dev_score = evaluator.evaluate(model, devset)

            metrics.append({
                'epoch': e,
                'train': train_metrics,
                'dev': dev_metrics
            })

            # Save
            if dev_score > best_dev_score:
                print('Setting best model after iteration', e)
                best_dev_score = dev_score
                best_dev_metrics = dev_metrics
                best_epoch = e
                best_num_samples = 0
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                tokenizer.save_pretrained(save_path)
                model.save_pretrained(save_path)

        # Store metrics
        with codecs.open(os.path.join(os.path.join(self.model_dir, dest), 'metrics.json'), 'w', encoding='utf-8') as f_out:
            json.dump({
                'best': {'epoch': best_epoch, 'metrics': best_dev_metrics, 'samples': best_num_samples},
                'epochs': metrics
            }, f_out)