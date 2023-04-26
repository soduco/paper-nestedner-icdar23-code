import numpy as np
import pandas as pd
import nltk
from xml.dom.minidom import parseString
from datasets import load_metric, load_from_disk, Dataset, DatasetDict
from config import logger
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed
)

LABELS_ID = {
    "O+O" : 0,
    "I-b_PER+O" : 1,
    "I-i_PER+O" : 2,
    "I-b_PER+b_TITREH" : 3,
    "I-i_PER+b_TITREH" : 4,
    "I-i_PER+i_TITREH" : 5,
    "I-b_ACT+O" : 6,
    "I-i_ACT+O" : 7,
    "I-b_DESC+O" : 8,
    "I-i_DESC+O" : 9,
    "I-b_DESC+b_ACT" : 10,
    "I-i_DESC+b_ACT" : 11,
    "I-i_DESC+i_ACT" : 12,
    "I-b_DESC+b_TITREP" : 13,
    "I-i_DESC+b_TITREP" : 14,
    "I-i_DESC+i_TITREP" : 15,
    "I-b_SPAT+O" : 16,
    "I-i_SPAT+O" : 17,
    "I-b_SPAT+b_LOC" : 18,
    "I-i_SPAT+b_LOC" : 19,
    "I-i_SPAT+i_LOC" : 20,
    "I-b_SPAT+b_CARDINAL" : 21,
    "I-i_SPAT+b_CARDINAL" : 22,
    "I-i_SPAT+i_CARDINAL" : 23,
    "I-b_SPAT+b_FT" : 24,
    "I-i_SPAT+b_FT" : 25,
    "I-i_SPAT+i_FT" : 26,
    "I-b_TITRE+O" : 27,
    "I-i_TITRE+O" : 28
}

#Lists for eval
##L1+L2
LABELS_ID_TO_L1L2 = ['O+O', 'B-O+PER', 'I-O+PER', 'B-TITREH+PER', 'B-TITREH+PER', 'I-TITREH+PER', 
                     'B-O+ACT', 'I-O+ACT', 'B-O+DESC', 'I-O+DESC', 'B-ACT+DESC', 'B-ACT+DESC', 'I-ACT+DESC','B-TITREP+DESC',
                     'B-TITREP+DESC', 'I-TITREP+DESC', 
                     'B-SPAT+O', 'I-SPAT+O','B-LOC+SPAT', 'B-LOC+SPAT','I-LOC+SPAT', 'B-CARDINAL+SPAT','B-CARDINAL+SPAT',
                     'I-CARDINAL+SPAT', 'B-FT+SPAT', 'B-FT+SPAT', 'I-FT+SPAT', 
                     'B-O+TITRE', 'I-O+TITRE']

##L1
LABELS_ID_TO_L1 = ["O","B-PER","I-PER","B-PER","I-PER","I-PER",
                  "B-ACT","I-ACT","B-DESC","I-DESC","B-DESC","I-DESC","I-DESC","B-DESC","I-DESC","I-DESC",
                  "B-SPAT","I-SPAT","B-SPAT","I-SPAT","I-SPAT","B-SPAT","I-SPAT","I-SPAT","B-SPAT","I-SPAT","I-SPAT",
                  "B-TITRE","I-TITRE"]

##L2
LABELS_ID_TO_L2 = ['O', 'O', 'O', 'B-TITREH', 'B-TITREH', 'I-TITREH', 'O', 'O', 'O', 'O', 'B-ACT', 'B-ACT', 'I-ACT', 'B-TITREP', 'B-TITREP', 'I-TITREP', 'O', 'O', 'B-LOC', 'B-LOC', 'I-LOC', 'B-CARDINAL', 'B-CARDINAL', 'I-CARDINAL', 'B-FT', 'B-FT', 'I-FT', 'O', 'O']

##DAS
LABELS_ID_TO_DAS = ['O','I-PER','I-PER','I-TITRE','I-TITRE','I-TITRE','I-ACT','I-ACT','O','O','I-ACT','I-ACT','I-ACT','I-TITRE','I-TITRE','I-TITRE','O','O','I-LOC','I-LOC','I-LOC','I-CARDINAL','I-CARDINAL','I-CARDINAL','I-FT','I-FT','I-FT','I-TITRE','I-TITRE']

# Entry point
def init_model(model_name, training_config,num_run:int):
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

# Metrics
def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = list(LABELS_ID.keys())
    
    ########################### Predicted labels ############################
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
    
    ########################### Cross entities L1+L2 ############################
    l1l2_predictions = []
    for entry in true_predictions:
        l1l2_entry = []
        for elem in entry:
            l1l2_label = LABELS_ID_TO_L1L2[LABELS_ID[elem]]
            l1l2_entry.append(l1l2_label)
        l1l2_predictions.append(l1l2_entry)
    
    l1l2_labels = []
    for entry in true_labels:
        l1l2_entry = []
        for elem in entry:
            l1l2_label = LABELS_ID_TO_L1L2[LABELS_ID[elem]]
            l1l2_entry.append(l1l2_label)
        l1l2_labels.append(l1l2_entry)
        
    metric_l1l2 = load_metric("seqeval")
    results_l1l2 = metric.compute(predictions=l1l2_predictions, references=l1l2_labels)
    
    ########################### Level-1 ############################
    l1_predictions = []
    for entry in true_predictions:
        l1_entry = []
        for elem in entry:
            l1_label = LABELS_ID_TO_L1[LABELS_ID[elem]]
            l1_entry.append(l1_label)
        l1_predictions.append(l1_entry)
    
    l1_labels = []
    for entry in true_labels:
        l1_entry = []
        for elem in entry:
            l1_label = LABELS_ID_TO_L1[LABELS_ID[elem]]
            l1_entry.append(l1_label)
        l1_labels.append(l1_entry)

    metric_l1 = load_metric("seqeval")
    results_l1 = metric_l1.compute(predictions=l1_predictions, references=l1_labels)

    ########################### Level-2 ############################
    l2_predictions = []
    for entry in true_predictions:
        l2_entry = []
        for elem in entry:
            l2_label = LABELS_ID_TO_L2[LABELS_ID[elem]]
            l2_entry.append(l2_label)
        l2_predictions.append(l2_entry)
    
    l2_labels = []
    for entry in true_labels:
        l2_entry = []
        for elem in entry:
            l2_label = LABELS_ID_TO_L2[LABELS_ID[elem]]
            l2_entry.append(l2_label)
        l2_labels.append(l2_entry)

    metric_l2 = load_metric("seqeval")
    results_l2 = metric_l2.compute(predictions=l2_predictions, references=l2_labels)
    
    ########################### DAS ############################
    das_predictions = []
    for entry in true_predictions:
        das_entry = []
        for elem in entry:
            das_label = LABELS_ID_TO_DAS[LABELS_ID[elem]]
            das_entry.append(das_label)
        das_predictions.append(das_entry)
        
    das_labels = []
    for entry in true_labels:
        das_entry = []
        for elem in entry:
            das_label = LABELS_ID_TO_DAS[LABELS_ID[elem]]
            das_entry.append(das_label)
        das_labels.append(das_entry)

    metric_das = load_metric("seqeval")
    results_das = metric_das.compute(predictions=das_predictions, references=das_labels)
    
    ########################### All ############################
    all_preds = l1_predictions + l2_predictions
    all_labels = l1_labels + l2_labels
    
    metric_all = load_metric("seqeval")
    results_all = metric_all.compute(predictions=all_preds, references=all_labels)
    
    json = {
        #Predicted labels
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        #L1
        "precision-l1": results_l1["overall_precision"],
        "recall-l1": results_l1["overall_recall"],
        "f1-l1": results_l1["overall_f1"],
        "accuracy-l1": results_l1["overall_accuracy"],
        #L2
        "precision-l2": results_l2["overall_precision"],
        "recall-l2": results_l2["overall_recall"],
        "f1-l2": results_l2["overall_f1"],
        "accuracy-l2": results_l2["overall_accuracy"],
        #L2
        "precision-das": results_das["overall_precision"],
        "recall-das": results_das["overall_recall"],
        "f1-das": results_das["overall_f1"],
        "accuracy-das": results_das["overall_accuracy"],
        #L1+L2
        "precision-l1l2": results_l1l2["overall_precision"],
        "recall-l1l2": results_l1l2["overall_recall"],
        "f1-l1l2": results_l1l2["overall_f1"],
        "accuracy-l1l2": results_l1l2["overall_accuracy"],
        #L1 and L2
        "precision-all": results_all["overall_precision"],
        "recall-all": results_all["overall_recall"],
        "f1-all": results_all["overall_f1"],
        "accuracy-all": results_all["overall_accuracy"],
            
        #By class
        "PER": results_all["PER"],
        "ACT": results_all["ACT"],
        "ACT_L1": results_l1["ACT"],
        "ACT_L2": results_l2["ACT"],
        "DESC": results_all["DESC"],
        "TITREH": results_all["TITREH"],
        "TITREP": results_all["TITREP"],
        "SPAT": results_all["SPAT"],
        "LOC": results_all["LOC"],
        "CARDINAL": results_all["CARDINAL"],
        "FT": results_all["FT"]
    }
    
    if 'TITRE' in list(results_all.keys()):
        json["TITRE"] = results_all["TITRE"]
        
    #Save reference labels and predictions
    #json["predictions"] = f"{true_predictions}"
    #json["labels"] = f"{true_labels}"
    
    return json
# =============================================================================
# region ~ Data conversion utils for Hugginface


#Model tokenizer
_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
# Models : camembert-base     Jean-Baptiste/camembert-ner    xlm-roberta-base     flaubert/flaubert_base_cased   HueyNemud/das22-10-camembert_pretrained

# convenient word tokenizer to create IOB-like data for the BERT models
nltk.download("punkt")

def create_huggingface_dataset_nested_ner(entries):
    # Creates a Huggingface Dataset from a set of NER-XML entries
    tokenized_entries = [word_tokens_from_nested_xml_iob2(entry) for entry in entries]
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

############################### Create labels from XML ##############################

#Not used for training Joint-labels to learn (classe IOB prefixes)

def word_tokens_from_nested_xml_iob2(entry):
    """
    Joint-labelling tokens from xml (2 levels) with IOB2 format
    """
    former_n1 = ''
    former_n2 = ''
    cat1_is_start = 0 # 0 = None 'O',1 = True,2 = False
    cat2_is_start = 0 # 0 = None 'O',1 = True,2 = False
    w_tokens = []
    labels = []
    
    entry_xml = f"<x>{entry}</x>"
    x = parseString(entry_xml).getElementsByTagName("x")[0]
    cat = ''
    ############ Level 1 ###############
    for el in x.childNodes:
        #If Outside
        if el.nodeName == "#text":
            cat = "O+O"
            txt = el.nodeValue
            words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
            w_tokens += words
            labels += [cat] * len(words)
            former_n1,former_n2 = 'O','O'
            cat1_is_start = 0
        #If I or B
        else:
            if el.nodeName == former_n1:
                cat1_s = f"I-i_{el.nodeName}"
                cat1_is_start = 2 #Ce n'est pas le premier token de l'entité
            elif el.nodeName != former_n1:
                cat1_s = f"I-b_{el.nodeName}"
                cat1_is_start = 1 #C'est le premier token de l'entité
            cat1 = f"I-i_{el.nodeName}"
            former_n1 = el.nodeName
            ############ Level 2 ###############
            node_num = 0
            for e in el.childNodes:
                #S'il n'y a pas de label suivant
                if e.nodeName == "#text":
                    cat2 = 'O'
                    txt = e.nodeValue
                    former_n2 = 'O'
                    cat2_is_start = 0
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

                words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
                w_tokens += words
                #Prefixes
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
    
    return w_tokens, labels

#endregion

#####################################################################
################ Dataset ############################################

def unwrap(list_of_tuples2):
    return tuple(zip(*list_of_tuples2))

def file_name(*parts, separator="_"):
    parts_str = [str(p) for p in parts]
    return separator.join(filter(None,parts_str))

def save_dataset_iob2(output_dir, datasets, names, suffix=None):
    """Export all datasets in Huggingface native formats."""
    assert len(datasets) == len(names)
    
    hf_dic = {}
    for ds, name in zip(datasets, names):

        examples, _ = unwrap(ds)
        
        hf_dic[name] = create_huggingface_dataset_nested_ner(examples)

    bert_ds = DatasetDict(hf_dic)
    hf_file = output_dir / file_name("huggingface", suffix)
    bert_ds.save_to_disk(hf_file)