from transformers import AutoTokenizer
import glob
from operator import itemgetter
from datasets import Dataset, DatasetDict
import re
from util_commons import orderLabelsbyEntsLevel, unwrap, file_name

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

#_convert_tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
_convert_tokenizer = AutoTokenizer.from_pretrained("HueyNemud/das22-10-camembert_pretrained")

def createIOB2Spans(s,entities,spans):
    
    iob_entities, iob_spans = [], []
    
    for i in range(len(entities)):
        span = s[spans[i][0]:spans[i][1]] #Text span
        _sub_spans = re.split('[ ,-]', span) #Split using whitespace
        
        if len(_sub_spans) == 1: #No whitespace
            iob_entities.append('B-' + entities[i])
            iob_spans.append(spans[i])
            
        elif len(_sub_spans) > 1:
            iob_entities.append('B-' + entities[i])
            iob_spans.append([spans[i][0],spans[i][0]+len(_sub_spans[0])])
            iob_entities.append('I-' + entities[i])
            iob_spans.append([spans[i][0]+len(_sub_spans[0]),spans[i][1]])
    #print(iob_entities, iob_spans)
    return iob_entities, iob_spans


def createJointLabels(ents):
    uppern1,uppern2 = list(zip(*ents))
    n1, n2 = [], []
    for elem in uppern1:
        elem = elem.replace('B-','b_')
        elem = elem.replace('I-','i_')
        n1.append(elem)
    for elem in uppern2:
        elem = elem.replace('B-','b_')
        elem = elem.replace('I-','i_')
        n2.append(elem)
    former_n1 = ''
    former_n2 = ''
    n1_jl = []
    n2_jl = []
    jl = []
    for e in n1:
        if e != former_n1 and e != "O":
            l = e.replace('B-','b_')
            n1_jl.append(l)
            former_n1 = e
        elif e == former_n1 and e != "O":
            l = e.replace('I-','i_')
            n1_jl.append(l)
            former_n1 = e
        else:
            n1_jl.append(e)
            former_n1 = e
            
    for f in n2:
        if f != former_n2 and f != "O":
            l = f.replace('B-','b_')
            n2_jl.append(l)
            former_n2 = f
        elif f == former_n2 and e != "O":
            l = f.replace('I-','i_')
            n2_jl.append(l)
            former_n2 = f
        else:
            n2_jl.append(f)
            former_n2 = f
    
    for i in range(len(n1)):
        if n1_jl[i] != 'O':
            jl.append('I-' + n1_jl[i] + '+' + n2_jl[i])
        else:
            jl.append(n1_jl[i] + '+' + n2_jl[i])
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

def create_huggingface_dataset_nested_ner_iob2(DATA):
    all_ents, all_tokens = [],[]
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

        entities, spans = createIOB2Spans(txt,entities,spans)

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
    
def save_dataset_iob2(output_dir, datasets, names, suffix=None):
    """Export all datasets in Huggingface native formats."""
    assert len(datasets) == len(names)
              
    hf_dic = {}
    for ds, name in zip(datasets, names):

        examples, _ = unwrap(ds)
        
        hf_dic[name],entries = create_huggingface_dataset_nested_ner_iob2(examples)
        if name == "test":
            with open(output_dir / "test_entries.txt",'w',encoding='utf-8') as f:
                for entry in entries:
                    f.write(entry + '\n')

    bert_ds = DatasetDict(hf_dic)
    hf_file = output_dir / file_name("huggingface", suffix)
    bert_ds.save_to_disk(hf_file)