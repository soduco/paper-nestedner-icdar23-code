from transformers import AutoTokenizer
import glob
from operator import itemgetter
from datasets import Dataset, DatasetDict
import re
from util_commons import orderLabelsbyEntsLevel, unwrap, file_name
import json

############################### Labels information ########################

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

# =============================================================================

#_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
#_convert_tokenizer = AutoTokenizer.from_pretrained("HueyNemud/das22-10-camembert_pretrained")
_convert_tokenizer = AutoTokenizer.from_pretrained("pjox/dalembert", add_prefix_space=True)

def createIOSpans(s,entities,spans):
    iob_entities, iob_spans = [], []
    
    for i in range(len(entities)):
        span = s[spans[i][0]:spans[i][1]] #Text span
        _sub_spans = re.split('[ ,-]', span) #Split using whitespace
        
        if len(_sub_spans) > 1:
            iob_entities.append('I-' + entities[i])
            iob_spans.append([spans[i][0],spans[i][0]+len(_sub_spans[0])])
            iob_entities.append('I-' + entities[i])
            iob_spans.append([spans[i][0]+len(_sub_spans[0]),spans[i][1]])
        else:
            iob_entities.append('I-' + entities[i])
            iob_spans.append(spans[i])
    print(iob_entities, iob_spans)
    return iob_entities, iob_spans


def createJointLabels(ents):
    n1,n2 = list(zip(*ents))
    jl = []
    for i in range(len(n1)):
        l = n2[i].replace('I-','i_')
        jl.append(n1[i] + '+' + l)
    return jl

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

def create_huggingface_dataset_nested_ner_io(DATA):
    all_ents, all_tokens = [],[]
    entries = []

    for i in range(len(DATA)):
        txt = DATA[i][0]
        ann = DATA[i][1:]
        entries.append(txt)
        all_ann = []
        for a in ann:
            all_ann.append([a[1],[a[3],a[4]]])
            print([a[1],[a[3],a[4]]])
        token_dict = _convert_tokenizer(txt)
        entities, spans = list(zip(*all_ann))

        entities, spans = createIOSpans(txt,entities,spans)
        tokens_entry, ents_entry = [], []

        for ix in range(1, len(token_dict["input_ids"]) - 1):
            token_id = token_dict["input_ids"][ix]
            token = _convert_tokenizer.convert_ids_to_tokens(token_id)
            cspan = token_dict.token_to_chars(ix)
            
            shift_on_space = lambda: int(ix - 1 > 0 and token[0] == "▁" and token != "▁")
            shifted_cspan = [cspan.start + shift_on_space(), cspan.end]

            ents = [ent for ix, ent in enumerate(entities) if
                    shifted_cspan[0] >= spans[ix][0] and shifted_cspan[1] - 1 < spans[ix][1]]

            ents_bylevel = orderLabelsbyEntsLevel(ents)
            #print(token, s[shifted_cspan[0]:shifted_cspan[1]], ents_bylevel)

            tokens_entry.append(token)
            ents_entry.append(ents_bylevel)
        labels = createJointLabels(ents_entry)

        all_tokens.append(tokens_entry)
        all_ents.append(labels)

    ds = Dataset.from_dict({"tokens": all_tokens, "ner_tags": all_ents})
    return ds.map(assign_labels_to_bert_tokens, batched=True), entries

############################### Save Dataset #####################################

def save_dataset_io(output_dir, datasets, names, suffix=None):
    """Export all datasets in Huggingface native formats."""
    assert len(datasets) == len(names)
              
    hf_dic = {}
    for ds, name in zip(datasets, names):

        examples, _ = unwrap(ds)
        
        hf_dic[name], entries = create_huggingface_dataset_nested_ner_io(examples)
        if name == "test":
            with open(output_dir / "test_entries.txt",'w',encoding='utf-8') as f:
                for entry in entries:
                    f.write(entry + '\n')

    bert_ds = DatasetDict(hf_dic)
    hf_file = output_dir / file_name("huggingface", suffix)
    bert_ds.save_to_disk(hf_file)