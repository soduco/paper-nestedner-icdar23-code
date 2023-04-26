import numpy as np
import nltk
from xml.dom.minidom import parseString
from datasets import load_metric, Dataset, DatasetDict
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

#Label ID
LABELS_ID = {
    "O+O" : 0,
    "I-PER+O" : 1,
    "I-PER+i_TITREH" : 2,
    "I-ACT+O" : 3,
    "I-DESC+O" : 4,
    "I-DESC+i_ACT" : 5,
    "I-DESC+i_TITREP" : 6,
    "I-SPAT+O" : 7,
    "I-SPAT+i_LOC" : 8,
    "I-SPAT+i_CARDINAL" : 9,
    "I-SPAT+i_FT" : 10,
    "I-TITRE+O" : 11
}

#Lists use for mapping to get by level results
LABELS_ID_TO_L1 = ["O","I-PER","I-PER","I-ACT","I-DESC","I-DESC","I-DESC","I-SPAT","I-SPAT","I-SPAT","I-SPAT","I-TITRE"]
LABELS_ID_TO_L2 = ["O","O","I-TITREH","O","O","I-ACT","I-TITREP","O","I-LOC","I-CARDINAL","I-FT","O"]
LABELS_ID_TO_DAS =["O","I-PER","I-TITRE","I-ACT","O","I-ACT","I-TITRE","O","I-LOC","I-CARDINAL","I-FT","I-TITRE"]

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
            args=training_args,
            train_dataset=train,
            eval_dataset=dev,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=patience)
            ]
        )

    trainer.train()
    return trainer.evaluate(test), trainer.evaluate()

# Metrics
def compute_metrics(p):
    
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    label_list = list(LABELS_ID.keys())
    
    #L1+L2
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
    
    #Level-1
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

    #Level-2
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
    
    #DAS
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
    
    #Global
    all_preds = l1_predictions + l2_predictions
    all_labels = l1_labels + l2_labels
    
    metric_all = load_metric("seqeval")
    results_all = metric_all.compute(predictions=all_preds, references=all_labels)
    
    json = {
        #Labels IO used for train and L1+L2 (same score)
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        
        "precision-l1": results_l1["overall_precision"],
        "recall-l1": results_l1["overall_recall"],
        "f1-l1": results_l1["overall_f1"],
        "accuracy-l1": results_l1["overall_accuracy"],
        
        "precision-l2": results_l2["overall_precision"],
        "recall-l2": results_l2["overall_recall"],
        "f1-l2": results_l2["overall_f1"],
        "accuracy-l2": results_l2["overall_accuracy"],
        
        "precision-all": results_all["overall_precision"],
        "recall-all": results_all["overall_recall"],
        "f1-all": results_all["overall_f1"],
        "accuracy-all": results_all["overall_accuracy"],
       
        "precision-das": results_das["overall_precision"],
        "recall-das": results_das["overall_recall"],
        "f1-das": results_das["overall_f1"],
        "accuracy-das": results_das["overall_accuracy"],
        
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
#_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
_convert_tokenizer = AutoTokenizer.from_pretrained("HueyNemud/das22-10-camembert_pretrained")

# convenient word tokenizer to create IOB-like data for the BERT models
nltk.download("punkt")

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
                print(label[word_id])
                label_id = LABELS_ID[label[word_id]]
                labels_ids.append(label_id)
            prev_word_id = word_id

        labels.append(labels_ids)

    bert_tokens["labels"] = labels
    return bert_tokens

def word_tokens_from_nested_xml(entry):
    """
    Joint-labelling tokens from xml (2 levels) with IO format string from the most fined-grained entity level
    """
    w_tokens = []
    labels = []
    entry_xml = f"<x>{entry}</x>"
    x = parseString(entry_xml).getElementsByTagName("x")[0]

    cat = ''
    for el in x.childNodes:
        if el.nodeName == "#text":
            cat = "O+O"
            txt = el.nodeValue
            #words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
            words = [txt]
            print(words)
            w_tokens += words
            print(w_tokens)
            labels += [cat] * len(words)
        else:
            cat1 = f"{el.nodeName}"
            for e in el.childNodes:
                if e.nodeName == "#text":
                    cat = 'I-' + cat1 + '+O'
                    txt = e.nodeValue
                else:
                    cat = f"I-" + cat1 + f"+i_{e.nodeName}"
                    txt = e.childNodes[0].nodeValue

                #words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
                words = [txt]
                print(words)
                w_tokens += words
                print(w_tokens)
                labels += [cat] * len(words)
    
    return w_tokens, labels

#endregion

#####################################################################
################ Dataset ############################################

def unwrap(list_of_tuples2):
    return tuple(zip(*list_of_tuples2))

def file_name(*parts, separator="_"):
    parts_str = [str(p) for p in parts]
    return separator.join(filter(None,parts_str))

def save_dataset_io(output_dir, datasets, names, suffix=None):
    """Export all datasets in Huggingface native formats."""
    assert len(datasets) == len(names)
    
    hf_dic = {}
    for ds, name in zip(datasets, names):

        examples, _ = unwrap(ds)
        
        hf_dic[name] = create_huggingface_dataset_nested_ner(examples)

    bert_ds = DatasetDict(hf_dic)
    hf_file = output_dir / file_name("huggingface", suffix)
    bert_ds.save_to_disk(hf_file)