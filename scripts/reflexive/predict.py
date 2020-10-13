import os

import torch
from matplotlib import pyplot
from transformers import BertTokenizer, BertForMaskedLM

# from: https://huggingface.co/transformers/quickstart.html#bert-example
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
import config

logging.basicConfig(level=logging.INFO)
GPU = True

# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(os.path.join(config.vocab_dirs['reflexive'], 'HighRefl_working'))

# Tokenize input
# correct = 'e19'
# text = '[CLS] e26 r0 [MASK] [SEP]'
# correct = 'e20'
# text = '[CLS] e9 r0 [MASK] [SEP]'
correct = 'e1'
text = '[CLS] e381 r7 [MASK] [SEP]'
tokenized_text = tokenizer.tokenize(text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_indices = [i for i, x in enumerate(tokenized_text) if x == "[MASK]"]
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0]*len(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Load pre-trained model (weights)
print('LOAD PRETRAINED BERTLM')
model = BertForMaskedLM.from_pretrained(os.path.join(config.output_dir, 'models', 'reflexive', 'HighRefl_working', 'checkpoint-800'))
model.eval()
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)


# If you have a GPU, put everything on cuda
if GPU:
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

for masked_index in masked_indices:
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)

    if predicted_token == correct:
        ranked_prediction = sorted(predictions[0, masked_index].cpu(), reverse=True)[:10]
        pyplot.plot(ranked_prediction)
        pyplot.plot()
        pyplot.show()
