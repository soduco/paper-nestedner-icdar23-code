# Code and Data for the paper "A Benchmark of Nested NER Approaches in Historical Structured Documents" presented at [ICDAR 2023](https://icdar2023.org/)

## Abstract
Named Entity Recognition (NER) is a key step in the creation of structured data from digitised historical documents. 
Traditional NER approaches deal with flat named entities, whereas entities often are nested. For example, a postal address might contain a street name and a number. This work compares three nested NER approaches, including two state-of-the-art approaches using Transformer-based architectures. We introduce a new Transformer-based approach based on joint labelling and semantic weighting of errors, evaluated on a collection of 19\textsuperscript{th}-century Paris trade directories. We evaluate approaches regarding the impact of supervised fine-tuning, unsupervised pre-training with noisy texts, and variation of IOB tagging formats.
Our results show that while nested NER approaches enable extracting structured data directly, they do not benefit from the extra knowledge provided during training and reach a performance similar to the base approach on flat entities. Even though all 3 approaches perform well in terms of F1 scores, joint labelling is most suitable for hierarchically structured data. Finally, our experiments reveal the superiority of the IO tagging format on such data.

## Sources documents
* Paper pre-print (PDF) : [![HAL - 03994759](https://img.shields.io/badge/HAL-03994759-38104A)](https://hal.science/hal-03994759) & [![arXiv](https://img.shields.io/badge/arXiv-2302.10204-b31b1b.svg)](https://arxiv.org/abs/2302.10204)
* Full dataset (images and transcripted texts) : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7864174.svg)](https://doi.org/10.5281/zenodo.7864174)

## Code
[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7997437.svg)](https://doi.org/10.5281/zenodo.7997437)

### Installation
```
pip install --requirement requirements.txt
```

### Models
* Source models [CamemBERT NER](https://huggingface.co/Jean-Baptiste/camembert-ner) and [CamemBERT NER pretrained on French trade directories](https://huggingface.co/HueyNemud/das22-10-camembert_pretrained) are shared on Huggingface Hub
* Paper's models and ready-to-load datasets [are shared on Huggingface Hub](https://huggingface.co/nlpso)

### Project Structure

Structure of this repository:

```
├── dataset                    <- Data used for training
│   ├── 10-ner_ref                <- Full ground-truth dataset
│   ├── 31-ner_align_pero         <- Full Pero-OCR dataset
│   ├── 41-ner_ref_from_pero      <- GT entries subset which have corresponding valid Pero OCR equivalent.
│
├── src                       <- Jupyter notebooks and Python scripts.
│   ├── global_metrics             <- Benchmark results tables
│   ├── m0_flat_ner                <- Flat NER approach notebook and scripts
│   ├── m1_independant_ner_layers  <- M1 approach notebook and scripts
|   ├── m2_joint-labelling_for_ner <- M2 approach notebook and scripts
│   ├── m3_hierarchical_ner        <- M3 approach notebook and scripts
|   ├── config.py
|   |── requirements.txt  
│
└── README.md
```
Please note that for each approach, the qualitative analysis notebook and the demo notebook can be run without preparing the source data neather training models.

## Reference
If you use this software, please cite it as below.
```
@inproceedings{nner_benchmark_2023,
	title = {A Benchmark of Nested Named Entity Recognition Approaches in Historical Structured Documents},
    author = {Tual, Solenn and Abadie, Nathalie and Carlinet, Edwin and Chazalon, Joseph and Duménieu, Bertrand},
    booktitle = {Proceedings of the 17th International Conference on Document Analysis and Recognition (ICDAR'23)},
    year = {2023},
    month = aug,
    address = {San José, California, USA},
	url = {https://hal.science/hal-03994759}
}
```

## Acknowledgment

This work is supported by the [French National Research Agency (ANR)](https://anr.fr/Projet-ANR-18-CE38-0013), as part of the [SODUCO project](https://soduco.github.io/) (grant ANR-18-CE38-0013).