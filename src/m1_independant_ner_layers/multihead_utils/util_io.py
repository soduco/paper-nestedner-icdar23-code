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
    "O": 0,          # Not an entity
    "I-PER": 1,      # A person, like "Baboulinet (Vincent)"
    "I-DESC": 2,      
    "I-TITRE": 3,    # A person's encoded title, like "O. ::LH::" for "Officier de la Légion d'Honneur"
    "I-SPAT": 4,       
    "I-TITREH": 5,       # A person's encoded title, like "O. ::LH::" for "Officier de la Légion d'Honneur"
    "I-TITREP": 6,       
    "I-ACT": 7,
    "I-LOC": 8,        # An address, like "rue du Faub. St.-Antoine"
    "I-CARDINAL":9,    # A street number
    "I-FT":10         # A feature type, like "fabrique" or "dépot" in front of addresses.
}

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


# Main loop
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
        
    return {
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
    tokenized_entries = [word_tokens_from_nested_xml_multihead(entry) for entry in entries]
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


def word_tokens_from_nested_xml_multihead(entry):
    """
    Joint-labelling tokens from xml (2 levels)
    """
    w_tokens = []
    labels_n1 = []
    labels_n2 = []
    entry_xml = f"<x>{entry}</x>"
    x = parseString(entry_xml).getElementsByTagName("x")[0]
    for el in x.childNodes:
        if el.nodeName == "#text":
            cat1 = "O"
            cat2 = "O"
            txt = el.nodeValue
            words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
            w_tokens += words
            labels_n1 += [cat1] * len(words)
            labels_n2 += [cat2] * len(words)

        else:
            cat1 = f"I-{el.nodeName}"
            for e in el.childNodes:
                if e.nodeName == "#text":
                    cat2 = "O"
                    txt = e.nodeValue
                else:
                    cat2 = f"I-{e.nodeName}"
                    txt = e.childNodes[0].nodeValue

                words = nltk.word_tokenize(txt, language="fr", preserve_line=True)
                w_tokens += words
                labels_n1 += [cat1] * len(words)
                labels_n2 += [cat2] * len(words)
    
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