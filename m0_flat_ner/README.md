# M0 : Flat NER

## Experiments

* **Experiment 1**
** Train ground-truth dataset using *CamemBERT* and *CamemBERT+ptrn* with IO labels

* **Experiment 2**
** Train noisy dataset using *CamemBERT* and *CamemBERT+ptrn* with IO labels

## Notebooks

* **00-prepare_datasets** : Create datasets object for Flat NER
* **10-experiment_1_ref** : Run experiment 1
* **20-experiment_2_pero_ocr** : Run experiment 2
* **30-experiments_1_2_figures** : Metrics
* **40_qualitative_analysis** : Qualitative analysis (loading models from HugginFace)
* **50-demo** : Demo (loading models from HugginFace)