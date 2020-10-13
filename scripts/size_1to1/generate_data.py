import argparse
import json
import os
import shutil
from collections import defaultdict
import config
from scripts.size_1to1 import datagen_config
import random

bert_config_1layer = {
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 12,
  "num_hidden_layers": 1,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 0
}

bert_config = {
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi",
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 192,
  "initializer_range": 0.02,
  "intermediate_size": 768,
  "max_position_embeddings": 512,
  "num_attention_heads": 3,
  "num_hidden_layers": 1,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3,
  "pooler_size_per_head": 128,
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2,
  "vocab_size": 0
}

def create_entities():
    return ['e' + str(i) for i in range(datagen_config.ENTITYTYPE_AMOUNT)]


def create_relations():
    return ['r' + str(i) for i in range(datagen_config.RELATIONTYPE_AMOUNT)]


def create_dataset(out_dir):
    triples = []
    eval_triples = []
    subject_rel2object = defaultdict(list)
    relations = create_relations()
    entities = create_entities()
    #idx_mid = int(len(entities)/2)
    #entities_sub = entities[0:idx_mid]
    #entities_obj = entities[idx_mid::]

    #for enitiy_sub, relation, entity_obj in zip(entities_sub, relations, entities_obj):
    for _ in range(datagen_config.NUM_FATS):
        enitiy_sub = random.choice(entities)
        relation = random.choice(relations)
        entity_obj = random.choice(entities)
        triple = f'{enitiy_sub} {relation} {entity_obj}'
        triples.append(triple)
        subject_rel2object[enitiy_sub + ' ' + relation].append(entity_obj)

    with open(os.path.join(out_dir, 'train.txt'), 'w') as dataset_file, \
            open(os.path.join(out_dir, 'eval.txt'), 'w') as eval_file:

        for triple in triples:
            dataset_file.write(triple + '\n')
        for triple in triples:
            eval_file.write(triple + '\n')

    with open(os.path.join(out_dir, 'subject_relation2object_eval.json'), 'w') as sub2obj_json:
        json.dump(subject_rel2object, sub2obj_json)
        with open(os.path.join(out_dir, 'subject_relation2object_train.json'), 'w') as sub2obj_json:
            json.dump(subject_rel2object, sub2obj_json)



def mask(dir, file):
    eval_path = os.path.join(dir, file)
    masked_path = os.path.join(dir, 'masked_' + file)

    with open(eval_path, 'r') as eval_file, open(masked_path, 'w') as masked_file:
        for line in eval_file:
            subj_rel = ' '.join(line.split()[:-1])
            masked = subj_rel + ' [MASK]\n'
            masked_file.write(masked)


def create_vocab(dir):
    freqs = defaultdict(int)
    with open(os.path.join(dir, 'train.txt'), 'r') as file:
        for line in file:
            words = line[:-1].split()
            for word in words:
                freqs[word] += 1

    vocab = {"[SEP]": 0, "[CLS]": 1, "[PAD]": 2, "[MASK]": 3, "[UNK]": 4}
    ranked = sorted(freqs.keys(), key=lambda x: freqs[x], reverse=True)
    ranked = dict([(val, idx+len(vocab)) for idx, val in enumerate(ranked)])
    path_to_vocab = os.path.join(config.vocab_dirs['size_1to1'], os.path.basename(dir))
    os.makedirs(path_to_vocab, exist_ok=True)
    with open(os.path.join(path_to_vocab, 'vocab.txt'), 'w') as txt_file:
        for key in vocab.keys():
            txt_file.write(key + '\n')
        for key in freqs.keys():
            txt_file.write(key + '\n')

    vocab.update(ranked)
    with open(os.path.join(path_to_vocab, 'vocab.json'), 'w') as json_file:
        json.dump(vocab, json_file)

    bert_config["vocab_size"] = len(vocab)
    with open(os.path.join(path_to_vocab, 'bert_config.json'), 'w') as json_file:
        json.dump(vocab, json_file)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", default=None, type=str, required=True, help="The name of the dataset you want to create."
    )
    args = parser.parse_args()
    DATA_DIR = os.path.join(config.datasets_dirs['size_1to1'], args.dataset_name)
    os.makedirs(DATA_DIR, exist_ok=False)
    create_dataset(DATA_DIR)
    mask(DATA_DIR, 'eval.txt')
    mask(DATA_DIR, 'train.txt')
    create_vocab(DATA_DIR)

    shutil.copy(config.datagen_config_size_1to1, os.path.join(DATA_DIR, 'datagen_config.py'))
