import json
import os
import shutil
from collections import defaultdict
from random import sample, shuffle
import numpy
import config
from scripts.transitive import datagen_config


class AdvancedTransitiveGenerator:

    def __init__(self, dataset_dir, config, evals_allowed_in_train=1, complete_pattern_per_line=True):
        self.dir = dataset_dir
        self.conf = config
        self.evals_allowed_in_train = evals_allowed_in_train
        self.complete_pattern_per_line = complete_pattern_per_line

        self.relations = ['r' + str(i) for i in range(self.conf.RELATIONTYPE_AMOUNT)]
        relations_split = int(self.conf.ratio_of_pattern_relations*len(self.relations))
        self.pattern_relations, self.random_relations = numpy.split(self.relations, [relations_split])
        self.entities = ['e' + str(i) for i in range(self.conf.ENTITYTYPE_AMOUNT)]

        self.subj_rel2obj_train, self.subj_rel2obj_eval = defaultdict(list), defaultdict(list)
        self.rand_subj_rel2obj_train, self.rand_subj_rel2obj_eval = defaultdict(list), defaultdict(list)
        self.accidental_rel2obj_eval, self.rand_accidental_rel2obj_eval = defaultdict(list), defaultdict(list)


        self.clear_files()

    def create_dataset(self):
        self.create_vocab()
        for _ in range(self.conf.NUMBER_RULES):
            relation = sample(list(self.pattern_relations), 1)[0]
            complete_facts = self.create_complete_facts(relation)
            split_pos = int(self.conf.ratio_of_complete_patterns * len(complete_facts))
            train, eval, hard_eval = self.split(complete_facts, split_pos)
            self.write(train, 'train', self.subj_rel2obj_train)
            self.write(eval, 'eval', self.subj_rel2obj_eval)
            general_patterns = self.create_complete_facts(relation, restriction=train, amount=40)
            accidental_pattern_eval, accidental_pattern_train = self.create_accidental_patterns(relation, restriction=train, amount=40)
            self.write(hard_eval, 'hard_eval', self.subj_rel2obj_eval)
            self.write(general_patterns, 'eval_general', self.subj_rel2obj_eval)
            self.write(accidental_pattern_eval, 'eval_wrong_pattern', self.accidental_rel2obj_eval)
            self.write(accidental_pattern_train, 'train', self.subj_rel2obj_train)
        for _ in range(self.conf.NUMBER_RULES):
            relation = sample(self.random_relations, 1)[0]
            rand_train, rand_eval, rand_hard_eval = self.create_incomplete_patterns(relation)
            self.write(rand_train, 'rand_train', self.rand_subj_rel2obj_train)
            self.write(rand_eval, 'rand_eval', self.rand_subj_rel2obj_eval)
            general_patterns = self.create_complete_facts(relation, restriction=rand_train, amount=40)
            accidental_pattern_eval, accidental_pattern_train = self.create_accidental_patterns(relation, restriction=rand_train, amount=40)
            self.write(rand_hard_eval, 'rand_hard_eval', self.rand_subj_rel2obj_eval)
            self.write(general_patterns, 'rand_eval_general', self.rand_subj_rel2obj_eval)
            self.write(accidental_pattern_eval, 'rand_eval_wrong_pattern', self.rand_accidental_rel2obj_eval)
            self.write(accidental_pattern_train, 'rand_train', self.rand_subj_rel2obj_train)

        json.dump(self.rand_subj_rel2obj_train, open(os.path.join(self.dir, 'rand_subject_relation2object_train.json'), 'w'))
        json.dump(self.rand_subj_rel2obj_eval, open(os.path.join(self.dir, 'rand_subject_relation2object_eval.json'), 'w'))
        json.dump(self.subj_rel2obj_train, open(os.path.join(self.dir, 'subject_relation2object_train.json'), 'w'))
        json.dump(self.subj_rel2obj_eval, open(os.path.join(self.dir, 'subject_relation2object_eval.json'), 'w'))
        json.dump(self.accidental_rel2obj_eval, open(os.path.join(self.dir, 'wrong_pattern_subject_relation2object_eval.json'), 'w'))
        json.dump(self.rand_accidental_rel2obj_eval, open(os.path.join(self.dir, 'rand_wrong_pattern_subject_relation2object_eval.json'), 'w'))

    def create_complete_facts(self, relation, restriction=None, amount=None):
        complete_facts = []
        if restriction is None:
            sampled = []
        else:
            if isinstance(restriction, numpy.ndarray):
                sampled = restriction.tolist()
            else:
                sampled = restriction
        if amount is None:
            amount = datagen_config.FACTS_PER_RELATION
        for _ in range(amount):
            a, b, c = sample(self.entities, 3)
            if {a, b, c} not in sampled:
                complete_facts.append(((a, relation, b), (b, relation, c), (a, relation, c)))
                sampled.append({a, b, c})
        return numpy.asarray(complete_facts)

    def create_incomplete_patterns(self, relation):
        train1 = []
        train2 = []
        eval = []
        # create incomplete patterns for evaluation during training
        for _ in range(datagen_config.FACTS_PER_RELATION//3):
            a, b, c = sample(self.entities, 3)
            if [[a, relation, b], [b, relation, c]] not in train1:
                train1.append([[a, relation, b], [b, relation, c]])
                eval.append([[a, relation, b], [b, relation, c], [a, relation, c]])
        # create complete random patterns
        for _ in range(datagen_config.FACTS_PER_RELATION//3):
            a, b, c, d, e, f = sample(self.entities, 6)
            if [[a, relation, b], [c, relation, d], [e, relation, f]] not in train2:
                train2.append([[a, relation, b], [c, relation, d], [e, relation, f]])
        # create patterns explicitly violating transitivity
        for _ in range(datagen_config.FACTS_PER_RELATION//3):
            a, b, c, d = sample(self.entities, 4)
            if [[a, relation, b], [b, relation, c], [a, relation, d]] not in train2:
                train2.append([[a, relation, b], [b, relation, c], [a, relation, d]])
        train = train1 + train2
        train1 = numpy.array(train1)
        train2 = numpy.array(train2)
        flatten_train1 = train1.reshape(train1.shape[0]*train1.shape[1], train1.shape[2])
        flatten_train2 = train2.reshape(train2.shape[0]*train2.shape[1], train2.shape[2])
        flatten_train = numpy.concatenate((flatten_train1, flatten_train2))
        eval = list(filter(lambda x: self.check_train(x[2], flatten_train.tolist(), 0), eval))
        hard_eval = [[x[2]] for x in eval]

        return numpy.asarray(train), numpy.asarray(eval), numpy.asarray(hard_eval)

    def clear_files(self):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)

    def create_vocab(self):
        vocab = ["[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]"] + self.relations + self.entities
        path_to_vocab = self.dir.replace('datasets', 'vocab')
        os.makedirs(path_to_vocab, exist_ok=True)
        with open(os.path.join(path_to_vocab, 'vocab.txt'), 'w') as txt_file:
            for voc in vocab:
                txt_file.write(voc + '\n')

    def split(self, facts, split_pos):
        train, eval = numpy.split(facts, [split_pos])
        flatten_train = train.reshape(train.shape[0]*train.shape[1], train.shape[2])
        train, to_be_masked = train.tolist(), eval.tolist()
        for chain in to_be_masked:
            train.append(chain[:-1])
        to_be_masked = list(filter(lambda x: self.check_train(x[2], flatten_train.tolist(), 1), to_be_masked))
        hard_to_be_masked = [[x[2]] for x in to_be_masked]
        return train, to_be_masked, hard_to_be_masked

    def write(self, triples, type, obj_dict):
        if 'train' in type and self.conf.shuffle_train:
            shuffle(triples)
        with open(os.path.join(self.dir, f'{type if type != "rand_train" else type.replace("rand_", "")}.txt'), 'a') as file, \
                open(os.path.join(self.dir, f'masked_{type}.txt'), 'a') as masked_file:
            for triple in triples:
                if 'train' in type:
                    for _ in range(numpy.random.randint(1, self.conf.MAX_INSTANCES_PER_FACT + 1)):
                        file.write(self.build_string(triple, mask=False))
                else:
                    file.write(self.build_string(triple, mask=False))
                if 'train' in type and self.conf.train_eval_random_subset:
                    if numpy.random.rand() < 0.1:
                        masked_file.write(self.build_string(triple, mask=True))
                else:
                    masked_file.write(self.build_string(triple, mask=True))
                key = self.build_string(triple, mask=True).replace(' [MASK]\n', '')
                obj_dict[key].append(triple[-1][-1] if self.complete_pattern_per_line else triple[-1])

    def build_string(self, array, mask):
        if self.complete_pattern_per_line:
            if mask:
                return ' '.join(' [SEP] '.join([' '.join(sub_array) for sub_array in array]).split()[:-1]) + ' [MASK]\n'
            else:
                return ' [SEP] '.join([' '.join(sub_array) for sub_array in array]) + '\n'
        else:
            if mask:
                return ' '.join(array[:-1]) + '\n'
            else:
                return ' '.join(array) + '\n'

    def check_train(self, x, train, minus):
        subj_rel = [fact[:-1] for fact in train]
        return subj_rel.count(x[:-1]) == self.evals_allowed_in_train - minus

    def create_accidental_patterns(self, relation, restriction, amount):
        complete_facts = []
        eval = []
        if isinstance(restriction, numpy.ndarray):
            restriction = restriction.tolist()
        for _ in range(amount):
            a, b, c, d = sample(self.entities, 4)
            if {a, b, c} not in restriction:
                complete_facts.append(((a, relation, b), (c, relation, d), (a, relation, d)))
                eval.append(((a, relation, b), (c, relation, d)))
                restriction.append({a, b, c})
        return numpy.asarray(complete_facts), numpy.asarray(eval)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_name", type=str, required=True, help="The name of the dataset you want to create.")
    # args = parser.parse_args()

    DATA_DIR = os.path.join(config.datasets_dirs['transitive'], 'StandardTrans')
    os.makedirs(DATA_DIR, exist_ok=False)
    generator = AdvancedTransitiveGenerator(DATA_DIR, datagen_config, complete_pattern_per_line=True)
    generator.create_dataset()

    shutil.copy(config.transitive_config, os.path.join(DATA_DIR, 'datagen_config.py'))
