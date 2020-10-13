import os
from transformers import BertForMaskedLM, BertTokenizer
import config
from scripts.groupings.predict import predict


masked_file = os.path.join(config.datasets_dirs['order'], 'StandardOrder_biggestLeftOut_variant_1400ents',
                           'masked_eval.txt')
correct_json_file = os.path.join(config.datasets_dirs['order'], 'StandardOrder_biggestLeftOut_variant_1400ents',
                                 'subject_relation2object_eval.json')

model = BertForMaskedLM.from_pretrained(os.path.join(config.output_dir, 'models', 'order',
                                                     'Standard_biggestLeftOut_withNumberPattern',
                                                     'checkpoint-32400'))
# Load pre-trained model tokenizer (vocabulary)
tokenizer = BertTokenizer.from_pretrained(os.path.join(config.vocab_dirs['order'],
                                                       'StandardOrder_biggestLeftOut_variant_1400ents'))

p = predict('order', model, tokenizer, masked_file, correct_json_file)
x = 5