import numpy as np
import nltk
from xml.dom.minidom import parseString
from datasets import load_metric, Dataset, DatasetDict
from config import logger
from multihead_utils.multihead_dataset_util import unwrap, file_name, train_dev_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed
)


LABELS_ID = {
    "O": 0,
    "I-PER": 1,
    "I-DESC": 2,
    "I-TITRE": 3,
    "I-SPAT": 4,
    "I-TITREH": 5,
    "I-TITREP": 6,
    "I-ACT": 7,
    "I-LOC": 8,
    "I-CARDINAL":9,
    "I-FT":10,
    "B-PER": 11,
    "B-DESC": 12,
    "B-TITRE": 13, 
    "B-SPAT": 14, 
    "B-TITREH": 15,
    "B-TITREP": 16,       
    "B-ACT": 17,
    "B-LOC": 18,
    "B-CARDINAL":19,
    "B-FT":20
}


# Entry point
def init_model(model_name, training_config,num_run):
    logger.info(f"Model {model_name}")
    
    set_seed(num_run)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True) 
    
    training_args = TrainingArguments(**training_config)

    # Load the model
    model = AutoModelForTokenClassification.from_pretrained( 
        model_name,
        num_labels=len(LABELS_ID),
        ignore_mismatched_sizes=True,
        id2label={v: k for k, v in LABELS_ID.items()},
        label2id=LABELS_ID
    )
    return model, tokenizer, training_args


# Main loop : entrainement
def train_eval_loop(model, training_args, tokenizer, train, dev, test, patience=5):

    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
            model,
            training_args,
            train_dataset=train,
            eval_dataset=dev,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=patience) 
            ],
        )
    
    trainer.train()
    return trainer.evaluate(test), trainer.evaluate()


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = list(LABELS_ID.keys())
    
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric = load_metric("seqeval")
    results = metric.compute(predictions=true_predictions, references=true_labels)
        
    return { #Sortie
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
         #Save reference labels and predictions to compute Global metrics and L1+L2 metrics in post-processing
        "predictions":f"{true_predictions}",
        "labels":f"{true_labels}"
    }

   
# =============================================================================
# region ~ Data conversion utils for Huggingface

#_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
_convert_tokenizer = AutoTokenizer.from_pretrained("HueyNemud/das22-10-camembert_pretrained")


def create_huggingface_dataset(entries):
    # Creates a Huggingface Dataset from a set of NER-XML entries
    tokenized_entries = [word_tokens_from_nested_xml_iob2(entry) for entry in entries]
    word_tokens, labels_n1, labels_n2 = zip(*tokenized_entries)
    ds = Dataset.from_dict({"tokens": word_tokens, 
                            "ner_tags_niv1": labels_n1,
                            "ner_tags_niv2": labels_n2,
                           })
    return ds.map(assign_labels_to_bert_tokens, batched=True)


def assign_labels_to_bert_tokens(examples):
    bert_tokens = _convert_tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    prev_word_id = None
    labels_n1 = []
    for id, label in enumerate(examples["ner_tags_niv1"]):
        labels_ids_n1 = []
        for word_id in bert_tokens.word_ids(batch_index=id):
            # Set the label only on the first token of each word
            if word_id in [None, prev_word_id]:
                labels_ids_n1.append(-100)
            else:
                label_id_n1 = LABELS_ID[label[word_id]]
                labels_ids_n1.append(label_id_n1)
            prev_word_id = word_id

        labels_n1.append(labels_ids_n1)
        
    prev_word_id = None
    labels_n2 = []
    for id, label in enumerate(examples["ner_tags_niv2"]):
        labels_ids_n2 = []
        for word_id in bert_tokens.word_ids(batch_index=id):
            # Set the label only on the first token of each word
            if word_id in [None, prev_word_id]:
                labels_ids_n2.append(-100)
            else:
                label_id_n2 = LABELS_ID[label[word_id]]
                labels_ids_n2.append(label_id_n2)
            prev_word_id = word_id

        labels_n2.append(labels_ids_n2)
     
    bert_tokens["labels_niv1"] = labels_n1
    bert_tokens["labels_niv2"] = labels_n2
    return bert_tokens


# convenient word tokenizer to create IOB-like data for the BERT models
nltk.download("punkt")


def word_tokens_from_nested_xml_iob2(entry):
    """
    Joint-labelling tokens from xml (2 levels) with BIO format
    """
    former_n1 = ''
    former_n2 = ''
    cat1_is_start = 0 # 0 = None 'O',1 = True,2 = False
    cat2_is_start = 0 # 0 = None 'O',1 = True,2 = False
    w_tokens = []
    labels = []
    #Création de la racine du xml
    entry_xml = f"<x>{entry}</x>"
    #Parser la chaîne xml
    x = parseString(entry_xml).getElementsByTagName("x")[0]
    cat = ''
    ############ Niveau 1 ###############
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
            former_n1,former_n2 = 'O','O'
            cat1_is_start = 0
        else:
            #La première partie du label correspond à cette catégorie
            if el.nodeName == former_n1:
                cat1_s = f"I-{el.nodeName}"
                cat1_is_start = 2
            elif el.nodeName != former_n1:
                cat1_s = f"B-{el.nodeName}"
                cat1_is_start = 1
            cat1 = f"I-{el.nodeName}"
            former_n1 = el.nodeName
            ############ Niveau 2 ###############
            node_num = 0
            for e in el.childNodes:
                #S'il n'y a pas de label suivant
                if e.nodeName == "#text":
                    cat2 = 'O'
                    txt = e.nodeValue
                    former_n2 = 'O'
                    cat2_is_start = 0
                    #print(txt + ' ==> ' + cat)
                #Pour tout autre label
                else:
                    if e.nodeName == former_n2 :
                        cat2_s = f"i_{e.nodeName}"
                        cat2_is_start = 2
                    elif e.nodeName != former_n2 :
                        cat2_s = f"b_{e.nodeName}"
                        cat2_is_start = 1
                    cat2 = f"i_{e.nodeName}"
                    txt = e.childNodes[0].nodeValue
                    former_n2 = e.nodeName
                    #print(txt + ' ==> ' + cat)

                words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
                w_tokens += words
                if node_num == 0 and cat2_is_start == 1:
                    cat = cat1_s + '+' + cat2_s
                    labels.append(cat)
                    cat = cat1 + '+' + cat2
                    labels += [cat] * (len(words)-1)
                elif node_num == 0 and cat2_is_start == 2:
                    cat = cat1_s + '+' + cat2
                    labels.append(cat)
                    cat = cat1 + '+' + cat2
                    labels += [cat] * (len(words)-1)
                elif node_num >= 1 and cat2_is_start == 1:
                    cat = cat1 + '+' + cat2_s
                    labels.append(cat)
                    cat = cat1 + '+' + cat2
                    labels += [cat] * (len(words)-1)
                elif node_num >= 1 and cat2_is_start == 2:
                    cat = cat1 + '+' + cat2
                    labels += [cat] * len(words)
                elif node_num == 0 and cat2_is_start == 0:
                    cat = cat1_s + '+' + 'O'
                    labels.append(cat)
                    cat = cat1 + '+' + 'O'
                    labels += [cat] * (len(words)-1)
                elif node_num >= 1 and cat2_is_start == 0:
                    cat = cat1 + '+' + 'O'
                    labels += [cat] * len(words)

                node_num += 1
              
    labels_n1 = []
    labels_n2 = []
    for i in range(len(labels)):
        labels_n1.append(labels[i].split('+')[0])
        n2 = labels[i].split('+')[1]
        n2 = n2.replace('i_','I-')
        n2 = n2.replace('b_','B-')
        labels_n2.append(n2)

    return w_tokens, labels_n1, labels_n2


# endregion

def save_dataset(output_dir, datasets, names, suffix=None):
    """Export all datasets in Huggingface native formats."""
    assert len(datasets) == len(names)
    
    hf_dic = {}
    for ds, name in zip(datasets, names):

        examples, _ = unwrap(ds)

        # Huggingface
        hf_dic[name] = create_huggingface_dataset(examples)

    bert_ds = DatasetDict(hf_dic)
    hf_file = output_dir / file_name("huggingface", suffix)
    bert_ds.save_to_disk(hf_file)