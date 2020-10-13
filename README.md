# Are Pretrained Language Models  Symbolic Reasoners over Knowledge?
This repository contains the code for ["Are Pretrained Language Models  Symbolic Reasoners over Knowledge?"](https://arxiv.org/pdf/2006.10413v2.pdf).

We provide a way to generate datasets that contain triples of the form "entity relation entity". These triplets follow different relational or logical rules.
We then train BERT from scratch on this data and evaluate its ability to generalize and these rules.
While we trained on BERT, this data generation process can in principle be used for testing any kind of language model.
The possible rules are:
- reflexivity
- symmetry
- inversion
- composition (and enhanced_composition)
- transitivity (and hierarchies+orders as structured versions of it)
- negation
- equivalence
- implication
- groupings

### Setup

We recommend running the following command in virtual environment:

    pip install -r requirements.txt

### Generating a dataset

To create your own dataset, navigate to `scripts/RELATION`. Here you can specify parameters (number of entities, number relations...) in `datagen_config.py`, which currently contain default parameters. Then run the following to create the data, here exemplified for symmetry:
python3 -m scripts.symmetry.generate_data --dataset_name MY_DATASET_NAME

The dataset will be written to `data/symmetry/datasets/MY_DATASET_NAME`.

### Training
To train the language model on a dataset, you run `run_language_modeling.py` as follows:
    
    python3 -m scripts.run_language_modeling \
    --relation RELATION_NAME
    --dataset_name DIR_NAME_OF_SPECIFIC_DATASET
    --anti SET_TRUE_IF_DATA_SHOULD_BE_EVALUATED_ON_ANTI_RULE_FACTS
    --random SET_TRUE_IF_DATA_SHOULD_BE_EVALUATED_ON_RANDOM_FACTS
    
where
 - `relation` is usually chosen from the above list of covered rules, e.g. "symmetry"; but you can come up with your own names as well
 - `DATA_DIR` When creating a dataset, you will have to specify a name. This name is then needed here.
 
Optional parameters:
- `anti` tells the script to look for a json with the answers. Only include this if the data has anti-rule relations.
- `random` tells the script to look for a json with the answers. Only include if the rule has relations with random facts.
- `numb_correct_answers` indicates how many correct answers a given query of the form "subject relation [MASK] has. This influences our metric for evaluation accuracy.
- `epochs`: number of epochs. Default is 2000
- `batch_size`: we recommmend the default 1024
- `learning_rate`: default is 6e-5
- many more parameters that we didn't change but could be changed in a model like BERT

The resulting model is saved under `outputs/model/RELATION/` and the events-file under `outputs/runs/RELATION/`.

Here is an exmaple command for symmetry:


    python3 -m scripts.run_language_modeling \
    --relation symmetry
    --dataset_name StandardSym
    --anti
    --random

### Probing BERT

We also provide our Notebooks and data for probing BERT for consistent predictions regarding symmetry & inversion.
This can be found under `probeBERT`.

## Citation
If you use this code, please cite:

      @misc{kassner2020pretrained,
            title={Are Pretrained Language Models Symbolic Reasoners Over Knowledge?}, 
            author={Nora Kassner and Benno Krojer and Hinrich Schütze},
            year={2020},
            eprint={2006.10413},
            archivePrefix={arXiv},
            primaryClass={cs.CL}
      }
