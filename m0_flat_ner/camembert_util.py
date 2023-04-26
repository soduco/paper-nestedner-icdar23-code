import numpy as np
import nltk
from xml.dom.minidom import parseString
from datasets import load_metric, Dataset
from config import logger
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    set_seed
)

#Label ID of original DAS code
LABELS_ID = {
    "O": 0,          # Not an entity
    "I-PER": 1,      # An address, like "rue du Faub. St.-Antoine"
    "I-TITRE": 2,      # A person, like "Baboulinet (Vincent)"
    "I-ACT": 3,      # Not used but present in the base model
    "I-LOC": 4, # Not used but present in the base model
    "I-CARDINAL": 5,      # An activity, like "plombier-devin"
    "I-FT": 6    # A person's encoded title, like "O. ::LH::" for "Officier de la Légion d'Honneur"
}

#To recreate flat reference using nested groundtruth
NESTED_LABELS_ID = {
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

FLAT_LABELS_ID = ['O','I-PER','I-TITRE','I-ACT','O','I-ACT','I-TITRE','O','I-LOC','I-CARDINAL','I-FT','I-TITRE']

# Entry point
def init_model(model_name, training_config,num_run):
    
    set_seed(num_run)
    
    logger.info(f"Model {model_name}")
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
    data_collator = DataCollatorForTokenClassification(tokenizer) #Permet de créer les batch à partir du dataset
    
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
        
    return { 
        #Global metrics
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        #By class
        "PER": results["PER"],
        "ACT": results["ACT"],
        "LOC": results["LOC"],
        "CARDINAL": results["CARDINAL"],
        "FT": results["FT"],
        "TITRE": results["TITRE"]
    }
   
# =============================================================================
# region ~ Data conversion utils for Hugginface

_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
#_convert_tokenizer = AutoTokenizer.from_pretrained("HueyNemud/das22-10-camembert_pretrained")

def create_huggingface_dataset(entries):
    # Creates a Huggingface Dataset from a set of NER-XML entries
    tokenized_entries = [word_tokens_from_xml(entry) for entry in entries]
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


# convenient word tokenizer to create IOB-like data for the BERT models
nltk.download("punkt")

def mappingNestedToFlat(labels,nestedlabels,flatlabels):
    flat_labels = []
    for label in labels:
        f_label = flatlabels[nestedlabels[label]]
        flat_labels.append(f_label)
    return flat_labels

def word_tokens_from_xml(entry):
    """
    Joint-labelling tokens from xml (2 levels) with IO format straing from the most fined-grained entity level
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
            words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
            w_tokens += words
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

                words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
                w_tokens += words
                labels += [cat] * len(words)
                
    #Align nested entities to flat ner entities 
    flat_labels = mappingNestedToFlat(labels,NESTED_LABELS_ID,FLAT_LABELS_ID)
    
    return w_tokens, flat_labels

# endregion
