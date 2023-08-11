import torch
import types
import numpy as np
import nltk
import random
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
    set_seed
)
from transformers.modeling_outputs import TokenClassifierOutput
from hierarchicalNER.hierarchicalLoss import HierarchicalCrossEntropyLoss
from hierarchicalNER.trees_IN_hmr import Ltree, Wtree

#Label ID of original DAS code

LABELS_ID = {
    "O+O+O+O" : 0,
    "I-geogFeat+O+O+O":1,
    "I-geogName+O+O+O":2,
    "I-geogName+i_name+O+O" : 3,
    "I-geogName+i_geogFeat+O+O" : 4,
    "I-geogName+i_geogName+i_name+O" : 5,
    "I-geogName+i_geogName+i_geogFeat+O" : 6,
    'I-geogName+i_geogName+O+O':7,
    'I-geogName+i_geogName+i_geogName+i_name':8,
    'I-geogName+i_geogName+i_geogName+i_geogFeat':9,
    'I-geogName+i_geogName+i_geogName+O':10
}

LABELS_ID_LIST = list(LABELS_ID.keys())

LABELS_ID_TO_L1 = ["O","I-geogFeat","I-geogName","I-geogName","I-geogName","I-geogName","I-geogName","I-geogName","I-geogName","I-geogName"]
LABELS_ID_TO_L2 = ["O","O","O","I-name","I-geogFeat","I-geogName","I-geogName","I-geogName","I-geogName","I-geogName"]
LABELS_ID_TO_L3 = ["O","O","O","O","O","I-name","I-geogFeat","O","I-geogName","I-geogName"]
LABELS_ID_TO_L4 = ["O","O","O","O","O","O","O","O","I-name","I-geogFeat"]

########################### Overwrite forward fuction of model ################

#Fixe new loss parameters
IGNORED_INDEX = -100
CustomLoss = HierarchicalCrossEntropyLoss(Ltree,LABELS_ID_LIST,Wtree, ignore_index=IGNORED_INDEX)

#Overwrite Token Classification classe forward
def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        #r
        #labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            #Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            #1]``.

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CustomLoss 
            #Originellement, CrossEntropyLoss : https://gist.github.com/mjdietzx/50d3c26f1fd543f1808ffffacc987cbf
            #Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
#############################################################################

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
    
    model.forward = types.MethodType(forward, model) #Monkey Patching
    return model, tokenizer, training_args

# Main loop
def train_eval_loop(model, training_args, tokenizer, train, dev, test, patience=3):
    
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
    
    #L1+L2+L3+L4
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
    
    #Level-3
    l3_predictions = []
    for entry in true_predictions:
        l3_entry = []
        for elem in entry:
            l3_label = LABELS_ID_TO_L3[LABELS_ID[elem]]
            l3_entry.append(l3_label)
        l3_predictions.append(l3_entry)
    
    l3_labels = []
    for entry in true_labels:
        l3_entry = []
        for elem in entry:
            l3_label = LABELS_ID_TO_L3[LABELS_ID[elem]]
            l3_entry.append(l3_label)
        l3_labels.append(l3_entry)

    metric_l3 = load_metric("seqeval")
    results_l3 = metric_l3.compute(predictions=l3_predictions, references=l3_labels)

    #Level-4
    l4_predictions = []
    for entry in true_predictions:
        l4_entry = []
        for elem in entry:
            l4_label = LABELS_ID_TO_L4[LABELS_ID[elem]]
            l4_entry.append(l4_label)
        l4_predictions.append(l4_entry)
    
    l4_labels = []
    for entry in true_labels:
        l4_entry = []
        for elem in entry:
            l4_label = LABELS_ID_TO_L4[LABELS_ID[elem]]
            l4_entry.append(l4_label)
        l4_labels.append(l4_entry)

    metric_l4 = load_metric("seqeval")
    results_l4 = metric_l4.compute(predictions=l4_predictions, references=l4_labels)

    #Global
    all_preds = l1_predictions + l2_predictions + l3_predictions + l4_predictions
    all_labels = l1_labels + l2_labels + l3_labels + l4_labels
    
    metric_all = load_metric("seqeval")
    results_all = metric_all.compute(predictions=all_preds, references=all_labels)
    
    json = {
        #Labels IO used for train and L1+L2+L3+L4 (same score)
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
        
        "precision-l3": results_l1["overall_precision"],
        "recall-l3": results_l1["overall_recall"],
        "f1-l3": results_l1["overall_f1"],
        "accuracy-l3": results_l1["overall_accuracy"],
        
        "precision-l4": results_l4["overall_precision"],
        "recall-l4": results_l4["overall_recall"],
        "f1-l4": results_l4["overall_f1"],
        "accuracy-l4": results_l4["overall_accuracy"],
        
        "precision-all": results_all["overall_precision"],
        "recall-all": results_all["overall_recall"],
        "f1-all": results_all["overall_f1"],
        "accuracy-all": results_all["overall_accuracy"],
        
        #By class
        "geogName": results_all["geogName"],
        "name": results_all["name"],
        "geogFeat": results_all["geogFeat"]
    }
    print(results_all)
    return json
   
# =============================================================================
# region ~ Data conversion utils for Hugginface

#Camembert without pretrain
_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")

def assign_labels_to_bert_tokens_IO(examples):
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