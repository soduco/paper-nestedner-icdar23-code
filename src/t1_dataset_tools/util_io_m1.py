from transformers import AutoTokenizer
import glob
from operator import itemgetter
from datasets import Dataset, DatasetDict
import re
from util_commons import orderLabelsbyEntsLevel, unwrap, file_name

############################### Labels information ########################

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
    "I-FT":10
}

# =============================================================================

#_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
_convert_tokenizer = AutoTokenizer.from_pretrained("HueyNemud/das22-10-camembert_pretrained")

def createIOSpans(s,entities,spans):
    
    iob_entities, iob_spans = [], []
    
    for i in range(len(entities)):
        span = s[spans[i][0]:spans[i][1]] #Text span
        _sub_spans = re.split('[ ,-]', span) #Split using whitespace
        
        if len(_sub_spans) == 1: #No whitespace
            iob_entities.append('I-' + entities[i])
            iob_spans.append(spans[i])
            
        elif len(_sub_spans) > 1:
            iob_entities.append('I-' + entities[i])
            iob_spans.append([spans[i][0],spans[i][0]+len(_sub_spans[0])])
            iob_entities.append('I-' + entities[i])
            iob_spans.append([spans[i][0]+len(_sub_spans[0]),spans[i][1]])

    return iob_entities, iob_spans

def assign_labels_to_bert_tokens(examples):
    bert_tokens = _convert_tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    prev_word_id = None
    labels_n1 = []
    for id, label in enumerate(examples["ner_tags_niv1"]):
        labels_ids_n1 = []
        for word_id in bert_tokens.word_ids(batch_index=id):
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


def create_huggingface_dataset_nested_ner_io(DATA):
    all_ents_l1, all_ents_l2, all_tokens = [],[],[]
    entries = []

    for i in range(len(DATA)):
        txt = DATA[i][0]
        ann = DATA[i][1:]
        entries.append(txt)
        all_ann = []
        for a in ann:
            all_ann.append([a[1],[a[3],a[4]]])

        token_dict = _convert_tokenizer(txt)
        entities, spans = list(zip(*all_ann))

        entities, spans = createIOSpans(txt,entities,spans)
        tokens_entry, ents_entry_l1, ents_entry_l2 = [], [], []

        for ix in range(1, len(token_dict["input_ids"]) - 1):
            token_id = token_dict["input_ids"][ix]
            token = _convert_tokenizer.convert_ids_to_tokens(token_id)
            cspan = token_dict.token_to_chars(ix)
            
            shift_on_space = lambda: int(ix - 1 > 0 and token[0] == "▁" and token != "▁")
            shifted_cspan = [cspan.start + shift_on_space(), cspan.end]

            ents = [ent for ix, ent in enumerate(entities) if
                    shifted_cspan[0] >= spans[ix][0] and shifted_cspan[1] - 1 < spans[ix][1]]

            ents_bylevel = orderLabelsbyEntsLevel(ents)
            tokens_entry.append(token)
            
            ents_entry_l1.append(ents_bylevel[0])
            ents_entry_l2.append(ents_bylevel[1])
            #print(token, s[shifted_cspan[0]:shifted_cspan[1]], ents_bylevel)

        all_tokens.append(tokens_entry)
        all_ents_l1.append(ents_entry_l1)
        all_ents_l2.append(ents_entry_l2)

    ds = Dataset.from_dict({"tokens": all_tokens, 
                        "ner_tags_niv1": all_ents_l1,
                        "ner_tags_niv2": all_ents_l2})
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