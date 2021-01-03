import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

# from: https://huggingface.co/transformers/quickstart.html#bert-example
# OPTIONAL: if you want to have more information on what's happening under the hood, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
GPU = False

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
text = '[CLS] a . [SEP] a bmw is faster than a vw .[SEP] a porsche is faster than a [MASK] . [SEP]'
tokenized_text = tokenizer.tokenize(text)
masked_indices = [i for i, x in enumerate(tokenized_text) if x == "[MASK]"]
print(tokenized_text)

indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
segments_ids = [0]*len(tokenized_text)

tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

print('LOAD PRETRAINED BERTLM')
model = BertForMaskedLM.from_pretrained('bert-large-uncased')
model.eval()

if GPU:
    tokens_tensor = tokens_tensor.to('cuda')
    segments_tensors = segments_tensors.to('cuda')
    model.to('cuda')

# Predict all tokens
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# confirm we were able to predict 'henson'
for masked_index in masked_indices:
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    print(predicted_token)
