# Code and Data for the paper "A Benchmark of Nested NER Approaches in Historical Documents" presented at [ICDAR 2023](https://icdar2023.org/)

## Sources documents
* Paper pre-print (PDF) : [HAL](https://hal.science/hal-03994759)
* Full dataset (images and transcripted texts) : [Zenodo](10.5281/zenodo.7864175)

## Code
[![python](https://img.shields.io/badge/-Python_3.9_%7C_3.10_%7C_3.11-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![DOI](https://zenodo.org/badge/632562117.svg)](https://zenodo.org/badge/latestdoi/632562117)

## Models
* Source models [CamemBERT NER](https://huggingface.co/Jean-Baptiste/camembert-ner) and [CamemBERT NER pretrained on French trade directories](https://huggingface.co/HueyNemud/das22-10-camembert_pretrained) are shared on Huggingface Hub
* Paper's models and ready-to-load datasets [are shared on Huggingface Hub](https://huggingface.co/nlpso)

## Project Structure

The directory structure of new project looks like this:

```
├── dataset                    <- Data used for training
│   ├── 10-ner_ref                <- Full ground-truth dataset
│   ├── 31-ner_align_pero         <- Full Pero-OCR dataset
│   ├── 41-ner_ref_from_pero      <- GT subset with valid Pero OCR equivalent.
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
├── src-latex                 <- Latex source code
│
├── rebuttal
│
└── README.md
```
