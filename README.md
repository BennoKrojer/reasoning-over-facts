# Are Pretrained Language Models  Symbolic Reasoners over Knowledge?
This repository contains the code for ["Are Pretrained Language Models  Symbolic Reasoners over Knowledge?"](https://arxiv.org/pdf/2006.10413v2.pdf).

We provide a way to generate datasets that contain triples of the form "entity relation entity". These triplets follow different symbolic rules.
We then train BERT from scratch on this data and evaluate its ability to generalize these rules.
While we trained on BERT, this data generation process can in principle be used for testing any kind of pre-trained language model.
We investigate the following RULES:
- equivalence
- symmetry
- inversion
- composition (and enhanced_composition)
- negation
- implication

### Setup

We recommend running the following command in virtual environment:

    pip install -r requirements.txt

### Generating a dataset

To create your own dataset, specify parameters (number of entities, relations, etc.) in `scripts/RULE/datagen_config.py`, then run the following to generate the dataset:
python3 -m scripts.RULE.generate_data --dataset_name MY_DATASET_NAME

The dataset will be written to `data/RULE/datasets/MY_DATASET_NAME`.

### Training
To train the language model on a dataset, you run `run_language_modeling.py` as follows:
    
    python3 -m scripts.run_language_modeling \
    --relation RELATION_NAME
    --dataset_name DIR_NAME_OF_SPECIFIC_DATASET
    
where
 - `relation` is usually chosen from RULES listed above, e.g. "symmetry"
 - `DATA_DIR` the name used to generate the dataset
 
Optional parameters:
-`gpu_device`: which gpu (default is 0) 
- `epochs`: number of epochs. (default is 2000)
- `batch_size`: number of samples per batch (default is 1024)
- `learning_rate`: default is 6e-5

The resulting model is saved under `outputs/model/RELATION/` and the events-file under `outputs/runs/RELATION/`.

Here is an exmaple command for symmetry:


    python3 -m scripts.run_language_modeling \
    --relation symmetry
    --dataset_name default_setup

### Probing BERT

We also provide our Notebooks and data for probing BERT for consistent predictions regarding symmetry & inversion.
This can be found under `probeBERT`.
For our bigger/smaller-than-probes, you can use `probeBERT_order.ibynb`.
For the other probes that just flipped subject and object, use `probeBERT_reverse.ipynb`.
For simple probing experimentation, use `probeBERT_simple.ipbynb`.

## Citation
If you use this code, please cite:

    @inproceedings{kassner-etal-2020-pretrained,
        title = "Are Pretrained Language Models Symbolic Reasoners over Knowledge?",
        author = {Kassner, Nora  and
          Krojer, Benno  and
          Sch{\"u}tze, Hinrich},
        booktitle = "Proceedings of the 24th Conference on Computational Natural Language Learning",
        month = nov,
        year = "2020",
        address = "Online",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/2020.conll-1.45",
        pages = "552--564",
        abstract = "How can pretrained language models (PLMs) learn factual knowledge from the training set? We investigate the two most important mechanisms: reasoning and memorization. Prior work has attempted to quantify the number of facts PLMs learn, but we present, using synthetic data, the first study that investigates the causal relation between facts present in training and facts learned by the PLM. For reasoning, we show that PLMs seem to learn to apply some symbolic reasoning rules correctly but struggle with others, including two-hop reasoning. Further analysis suggests that even the application of learned reasoning rules is flawed. For memorization, we identify schema conformity (facts systematically supported by other facts) and frequency as key factors for its success.",
    }
