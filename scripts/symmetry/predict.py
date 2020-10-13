import json
import os
from typing import List
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import BertTokenizer, BertForMaskedLM, PreTrainedTokenizer
import config


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)[
            "input_ids"]
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def collate(examples: List[torch.Tensor]):
    if tokenizer._pad_token is None:
        return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)


def get_mask_idx(batch):
    mask_token = tokenizer.mask_token_id
    return [list(batch[i]).index(mask_token) for i in range(batch.shape[0])]


def compute_ranked_accuracy(loader, json, relation, out_correct, out_false):
    accurate = 0
    total = 0
    for batch in loader:
        mask_idx = get_mask_idx(batch)
        prediction_scores = model(batch)[0]
        prediction_scores = prediction_scores[np.arange(prediction_scores.shape[0]), mask_idx, :]
        if relation == 'groupings':
            numb_ranked_predictions = 5  # TODO: pass this properly! 2 for transitive, 5 for groupings, relative for orders
        elif relation == 'transitive':
            numb_ranked_predictions = 2
        elif relation == 'order':
            numb_ranked_predictions = 5
        predicted_ids = torch.argsort(prediction_scores, dim=1, descending=True)[:, :numb_ranked_predictions]
        ranked_predictions = []
        for i in range(numb_ranked_predictions):
            ranked_predictions.append(tokenizer.convert_ids_to_tokens(predicted_ids[:, i]))
        predicted_strings = list(zip(*ranked_predictions))
        for i, (predicted, sample) in enumerate(zip(predicted_strings, batch)):
            key = " ".join(tokenizer.convert_ids_to_tokens(sample[1:mask_idx[i]]))
            correct_objects = json[key]
            if len(set(predicted[:numb_ranked_predictions]) & set(correct_objects)) > 0:
                accurate += 1
            total += 1
    return accurate / total


def compute_accuracy(loader, json, out_correct, out_false):
    accurate = 0
    total = 0
    for batch in loader:
        mask_idx = get_mask_idx(batch)
        prediction_scores = model(batch)[0]
        prediction_scores = prediction_scores[np.arange(prediction_scores.shape[0]), mask_idx, :]
        predicted_ids = torch.argmax(prediction_scores, dim=1)
        predicted_strings = tokenizer.convert_ids_to_tokens(predicted_ids)
        for i, (predicted, sample) in enumerate(zip(predicted_strings, batch)):
            key = " ".join(tokenizer.convert_ids_to_tokens(sample[1:mask_idx[i]]))
            correct_objects = json[key]
            if (predicted in correct_objects) if type not in ['rand_eval', 'anti_eval'] else (predicted not in
                                                                                              correct_objects):
                accurate += 1
                out_correct.write(f'{key} -> {predicted} | {correct_objects}\n')
            else:
                out_false.write(f'{key} -> {predicted} | {correct_objects}\n')
            total += 1
    return accurate / total


model = BertForMaskedLM.from_pretrained(
    os.path.join(config.output_dir, 'models', 'symmetry', 'StandardSym_600ep','checkpoint-68400'))
model.eval()

tokenizer = BertTokenizer.from_pretrained(os.path.join(config.vocab_dirs['symmetry'], 'StandardSym'))

masked_file = os.path.join(config.datasets_dirs['symmetry'], 'StandardSym', 'masked_rand_train.txt')
correct_json_file = os.path.join(config.datasets_dirs['symmetry'], 'StandardSym',
                              'rand_subject_relation2object_train.json')
correct_json = json.load(open(correct_json_file, 'r'))
eval_type = 'rand_train'
relation = 'symmetry'

result = {}
logging_predictions = {}

dataset = LineByLineTextDataset(tokenizer, file_path=masked_file, block_size=tokenizer.max_len)
dataloader = DataLoader(
    dataset, sampler=SequentialSampler(dataset), batch_size=256, collate_fn=collate)
out_correct = open('correct_preds', 'w')
out_false = open('false_preds', 'w')

with torch.no_grad():
    if relation in ['transitive', 'groupings', 'order'] and eval_type == 'eval':
        accuracy = compute_ranked_accuracy(dataloader, correct_json, relation, out_correct, out_false)
    else:
        accuracy = compute_accuracy(dataloader, correct_json, out_correct, out_false)
    print(accuracy)

