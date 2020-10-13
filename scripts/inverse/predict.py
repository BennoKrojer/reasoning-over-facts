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


def predict(model, tokenizer, masked_file, correct):

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    def get_mask_idx(batch, tokenizer):
        mask_token = tokenizer.mask_token_id
        return [list(batch[i]).index(mask_token) for i in range(batch.shape[0])]

    def compute_ranked_accuracy(loader, json, predictions):
        accurate = 0
        total = 0
        for batch in loader:
            mask_idx = get_mask_idx(batch, tokenizer)
            prediction_scores = model(batch)[0]
            prediction_scores = prediction_scores[np.arange(prediction_scores.shape[0]), mask_idx, :]
            for i, (predicted, sample) in enumerate(zip(prediction_scores, batch)):
                key = " ".join(tokenizer.convert_ids_to_tokens(sample[1:mask_idx[i]]))
                correct_objects = json[key]
                numb_correct_answers = len(correct_objects)
                predicted_ids = torch.argsort(predicted, dim=0, descending=True)[:numb_correct_answers]
                ranked_predictions = tokenizer.convert_ids_to_tokens(predicted_ids)
                accurate += len(set(ranked_predictions) & set(correct_objects))/numb_correct_answers
                total += numb_correct_answers
                predictions.append({'probe': key, 'predictions': ranked_predictions, 'correct': correct_objects})
        return accurate / total, predictions

    model.eval()
    correct_json = json.load(open(correct, 'r'))

    dataset = LineByLineTextDataset(tokenizer, file_path=masked_file, block_size=tokenizer.max_len)
    dataloader = DataLoader(
        dataset, sampler=SequentialSampler(dataset), batch_size=256, collate_fn=collate)
    predictions = []

    with torch.no_grad():
        accuracy, predictions = compute_ranked_accuracy(dataloader, correct_json, predictions)
        print(accuracy)
        json.dump(predictions, open('pred.json', 'w'), indent=2)
    return predictions


if __name__ == '__main__':
    model = BertForMaskedLM.from_pretrained(
        os.path.join(config.output_dir, 'models', 'inverse', 'StandardInv_with_anti', 'checkpoint-75000'))
    tokenizer = BertTokenizer.from_pretrained(os.path.join(config.vocab_dirs['inverse'],
                                                           'StandardInv_with_anti'))
    masked_file = os.path.join(config.datasets_dirs['inverse'], 'StandardInv_with_anti', 'masked_eval.txt')
    correct_json_file = os.path.join(config.datasets_dirs['inverse'], 'StandardInv_with_anti',
                                     'subject_relation2object_eval.json')
    predict(model, tokenizer, masked_file, correct_json_file)
