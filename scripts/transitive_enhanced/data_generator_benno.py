import json
import os
import shutil
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy
from numpy.random import shuffle
from random import sample


class DataGenerator(ABC):

    def __init__(self, dataset_dir, config, evals_allowed_in_train, complete_pattern_per_line=False,
                 numb_left_out_for_eval=1):
        self.dir = dataset_dir
        self.conf = config
        self.evals_allowed_in_train = evals_allowed_in_train
        self.complete_pattern_per_line = complete_pattern_per_line
        self.leave_out = numb_left_out_for_eval

        self.relations = ['r' + str(i) for i in range(self.conf.RELATIONTYPE_AMOUNT)]
        relations_split = int(self.conf.ratio_of_pattern_relations * len(self.relations))
        self.pattern_relations, self.random_relations = numpy.split(self.relations, [relations_split])
        self.entities = ['e' + str(i) for i in range(self.conf.ENTITYTYPE_AMOUNT)]
        self.attributes = ['a' + str(i) for i in range(int(len(self.entities)/self.groupsize))]
        self.subj_rel2obj_train, self.subj_rel2obj_eval = defaultdict(list), defaultdict(list)
        self.rand_subj_rel2obj_train, self.rand_subj_rel2obj_eval = defaultdict(list), defaultdict(list)
        self.group_entities = []
        self.clear_files()

    def create_dataset(self):
        self.create_vocab()
        # available_entities = self.entities.copy()
        # available_attributes = self.attributes.copy()

        # for _ in range(500):
        #     entities = sample(available_entities, self.groupsize)
        #     for e in entities:
        #         available_entities.remove(e)
        #     group_name = sample(available_attributes, 1)[0]
        #     available_attributes.remove(group_name)
        #     self.group_entities.append([entities, group_name])

        for _ in range(self.conf.NUMBER_RULES):
            relations = sample(list(self.pattern_relations), 3)
            complete_facts = self.create_complete_facts(relations)
            split_pos = int(self.conf.ratio_of_complete_patterns * len(complete_facts))
            train, eval = self.split(complete_facts, split_pos)
            self.write(train, 'train', self.subj_rel2obj_train)
            self.write(eval, 'eval', self.subj_rel2obj_eval)

        for _ in range(self.conf.NUMBER_RULES):
            relation = sample(list(self.random_relations), 1)[0]
            rand_train, rand_eval = self.create_incomplete_patterns(relation)
            self.write(rand_train, 'rand_train', self.rand_subj_rel2obj_train)
            self.write(rand_eval, 'rand_eval', self.rand_subj_rel2obj_eval)

        json.dump(self.rand_subj_rel2obj_train,
                  open(os.path.join(self.dir, 'rand_subject_relation2object_train.json'), 'w'))
        json.dump(self.rand_subj_rel2obj_eval,
                  open(os.path.join(self.dir, 'rand_subject_relation2object_eval.json'), 'w'))
        json.dump(self.subj_rel2obj_train, open(os.path.join(self.dir, 'subject_relation2object_train.json'), 'w'))
        json.dump(self.subj_rel2obj_eval, open(os.path.join(self.dir, 'subject_relation2object_eval.json'), 'w'))
        self.create_vocab()

    def clear_files(self):
        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)

    def create_vocab(self):
        vocab = ["[SEP]", "[CLS]", "[PAD]", "[MASK]", "[UNK]", "connectedto", 'is'] + self.relations + self.entities \
                + [a+i for i in ['x','y','z'] for a in self.attributes]
        path_to_vocab = self.dir.replace('datasets', 'vocab')
        os.makedirs(path_to_vocab, exist_ok=True)
        with open(os.path.join(path_to_vocab, 'vocab.txt'), 'w') as txt_file:
            for voc in vocab:
                txt_file.write(voc + '\n')

    @abstractmethod
    def create_complete_facts(self, relation):
        pass

    def split(self, facts, split_pos):
        if facts.shape[1] == 2:
            train = facts[:,0]
            eval = facts[:,1]
            train_part, eval = numpy.split(eval, [split_pos])
            train = numpy.concatenate([train, train_part])
        train, eval = numpy.split(facts, [split_pos])
        flatten_train = train.reshape(train.shape[0] * train.shape[1], train.shape[2])
        train = flatten_train
        train, eval = train.tolist(), eval.tolist()
        to_be_masked = []
        for chain in eval:
            train += chain[:-self.leave_out]
            to_be_masked.append(chain[-self.leave_out:])
        to_be_masked = list(filter(lambda x: self.check_train(x, train, 0), to_be_masked))
        to_be_masked = numpy.array(to_be_masked)
        to_be_masked = numpy.array(to_be_masked).reshape(to_be_masked.shape[0] * to_be_masked.shape[1],
                                                         to_be_masked.shape[2])
        return train, to_be_masked

    def write(self, triples, type, obj_dict):
        with open(os.path.join(self.dir, f'{type if "_train" not in type else "train"}.txt'),
                  'a') as file, open(os.path.join(self.dir, f'masked_{type}.txt'), 'a') as masked_file:

            masked_facts = dict()
            for triple in triples:
                if 'train' in type:
                    for _ in range(numpy.random.randint(1, self.conf.MAX_INSTANCES_PER_FACT + 1)):
                        file.write(self.build_string(triple, mask=False))
                else:
                    file.write(self.build_string(triple, mask=False))
                if 'train' in type and self.conf.train_eval_random_subset:
                    if numpy.random.rand() < 0.1:
                        masked_facts[self.build_string(triple, mask=True)] = None
                else:
                    masked_facts[self.build_string(triple, mask=True)] = None
                obj_dict[self.build_string(triple, mask=True).replace(' [MASK]\n', '')].append(
                    triple[-1][-1] if self.complete_pattern_per_line else triple[-1])
            for fact in masked_facts:
                masked_file.write(fact)

    def build_string(self, array, mask):
        if self.complete_pattern_per_line:
            if mask:
                return ' '.join(' [SEP] '.join([' '.join(sub_array) for sub_array in array]).split()[:-1]) + ' [MASK]\n'
            else:
                return ' [SEP] '.join([' '.join(sub_array) for sub_array in array]) + '\n'
        else:
            if mask:
                return ' '.join(array[:-1]) + ' [MASK]\n'
            else:
                return ' '.join(array) + '\n'

    def check_train(self, x, train, minus):
        subj_rel = [fact[:-1] for fact in train]
        return subj_rel.count(x[:-1]) == self.evals_allowed_in_train - minus

    @abstractmethod
    def create_incomplete_patterns(self, relation):
        pass
