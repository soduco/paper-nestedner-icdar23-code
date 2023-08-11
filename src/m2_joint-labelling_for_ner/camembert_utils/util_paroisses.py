import numpy as np
import nltk
from xml.dom.minidom import parseString
from datasets import load_metric, Dataset, DatasetDict
from config import logger
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

LABELS_ID = {
    'O+O':0,
    'I-date+O':1,
    'I-geogName+O':2,
    'I-geogName+i_featureType':3,
    'I-geogName+i_name':4,
    'I-geogName+i_place':5,
    'I-place+O':6,
    'I-place+i_featureType':7,
    'I-place+i_name':8,
}

def unwrap(list_of_tuples2):
    #Itérateur
    #retourne les résultats sous la forme de tuples
    return tuple(zip(*list_of_tuples2))

def file_name(*parts, separator="_"):
    parts_str = [str(p) for p in parts]
    return separator.join(filter(None,parts_str))

def init_model(model_name, training_config):
    """
    Init model function
    In :
     - model_name : name of the model load from transformers library
     - training_config : training params (eval steps, learnig rate etc)
     
     Out :
     - model : model loads from Transformers library without pre-training or fine-tuning ; with config and weights
     - tokenizer : 
     - training_args : 
    """
    logger.info(f"Model {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True) 
    #Chargement du tokenizer associé au modèle camembert d'HuggingFace
    
    #output_path = save_model_path or "/tmp/bert-model"    
    training_args = TrainingArguments(**training_config) #** en python : permet de transformer le dictionnaire clé-valeur défini à l'extérieur de la fonction en une liste d'aruments pour la fonction

    # Load the model
    #Charger un modèle pré entrainé de classification de token HuggingFace (ici ce sera CamemBERT)
    model = AutoModelForTokenClassification.from_pretrained( 
            #From_pretrained permet d'instancier les poids du modèle générique 
        model_name,
        num_labels=len(LABELS_ID),#Nombre de labels possibles
        ignore_mismatched_sizes=True,
        id2label={v: k for k, v in LABELS_ID.items()},
        label2id=LABELS_ID
    )
    return model, tokenizer, training_args

# Main loop : entrainement
def train_eval_loop(model, training_args, tokenizer, train, dev, test, patience=3):
    """
    In :
    - model : modèle chargé epuis Transformers
    - training_args : paramètres d'entrainement
    - tokenizer utilisé
    - jeu d'entrainement
    - jeu de développement
    - jeu de test
    - patience
    """
    data_collator = DataCollatorForTokenClassification(tokenizer) #Permet de créer les batch à partir du dataset
    #Outils d'entrainement
    trainer = Trainer(
            model, #Modèle chargé depuis Transformers
            training_args, #Paramètres d'entrainement
            train_dataset=train,
            eval_dataset=dev,
            data_collator=data_collator,#batchs
            tokenizer=tokenizer,#tokenizer
            compute_metrics=compute_metrics,#calcule les métriques avec la fonction dédiée
            #callbacks=[ #callback : prend des décisions en fonction de l'avancement de l'entrainement
                #EarlyStoppingCallback(early_stopping_patience=patience)#Arrête l'entrainement une fois que 
            #],
        )
    trainer.train() #Réalise l'entrainement
    return trainer.evaluate(test), trainer.evaluate() #Retourne le résultat de l'évaluation, appelle la fonction compute_metrics

#Compte metrics
def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2) #Retourne la valeur maximale sur un axe d'index fourni
    label_list = list(LABELS_ID.keys())
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval")#Chargement des métriques souhaitées
    results = metric.compute(predictions=true_predictions, references=true_labels)#Calcul des métriques
    print(results)
    return { #Sortie
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# =============================================================================
# region ~ Data conversion utils for Hugginface

_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")

def create_huggingface_dataset_nested_ner(entries):
    # Creates a Huggingface Dataset from a set of NER-XML entries
    tokenized_entries = [word_tokens_from_nested_xml(entry) for entry in entries]
    word_tokens, labels = zip(*tokenized_entries)
    ds = Dataset.from_dict({"tokens": word_tokens, "ner_tags": labels})
    return ds.map(assign_labels_to_bert_tokens, batched=True)


def assign_labels_to_bert_tokens(examples):
    bert_tokens = _convert_tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    prev_word_id = None
    labels = []
    for id, label in enumerate(examples["ner_tags"]):
        labels_ids = []
        for word_id in bert_tokens.word_ids(batch_index=id):
            # Set the label only on the first token of each word
            if word_id in [None, prev_word_id]:
                labels_ids.append(-100)
            else:
                label_id = LABELS_ID[label[word_id]]
                labels_ids.append(label_id)
            prev_word_id = word_id

        labels.append(labels_ids)

    bert_tokens["labels"] = labels
    return bert_tokens

nltk.download("punkt")

def word_tokens_from_nested_xml(entry):
    """
    Joint-labelling tokens from xml (2 levels) with IO format
    """
    w_tokens = []
    labels = []
    #Création de la racine du xml
    entry_xml = f"<x>{entry}</x>"
    #print(entry_xml)
    #Parser la chaîne xml
    x = parseString(entry_xml).getElementsByTagName("x")[0]
    #print(x)
    cat = ''
    for el in x.childNodes:
        #Si le texte ne se trouve dans aucune balise XML
        if el.nodeName == "#text":
            cat = "O+O"
            txt = el.nodeValue
            #print(txt + ' ==> ' + cat)
        #Si le texte se trouve dans une balise XML O+O
            words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
            w_tokens += words
            labels += [cat] * len(words)

        else:
            #La première partie du label correspond à cette catégorie
            cat1 = f"I-{el.nodeName}"
            #On cherche la seconde partie du label
            for e in el.childNodes:
                #S'il n'y a pas de label suivant
                if e.nodeName == "#text":
                    cat = cat1 + "+O"
                    txt = e.nodeValue
                    #print(txt + ' ==> ' + cat)
                #Pour tout autre label
                else:
                    cat = cat1 + f"+i_{e.nodeName}"
                    txt = e.childNodes[0].nodeValue
                    #print(txt + ' ==> ' + cat)

                words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
                w_tokens += words
                labels += [cat] * len(words)

    return w_tokens, labels


def save_dataset(output_dir, datasets, names, suffix=None):
    """Export all datasets in Huggingface native formats."""
    assert len(datasets) == len(names)
    
    hf_dic = {}
    for ds, name in zip(datasets, names):

        examples, _ = unwrap(ds)
        
        hf_dic[name] = create_huggingface_dataset_nested_ner(examples)

    bert_ds = DatasetDict(hf_dic)
    hf_file = output_dir / file_name("huggingface", suffix)
    bert_ds.save_to_disk(hf_file)