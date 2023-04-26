from lxml import etree
import pandas as pd
import re
import numpy as np

def get_NER_tags(entry,res,format_):
    """
    For one entry and its list of named entity, return entry in XML format
    Params:
        entry : entrée brute, sans annotations
        res (list of dict) : pipeline res with aggregation mode set to None
    """
    #Number of levels
    num_levels = res[0]["entity"].count('+') + 1

    all_types, all_words, all_starts, all_ends = [],[],[],[]

    #For each level
    for j in range(num_levels):
        #### Init var
        if res[0]["entity"] != "O+O":
            former_type = res[0]["entity"].split('+')[j][2:]
            former_prefix = res[0]["entity"].split('+')[j][:1]
        else:
            former_type, former_prefix = 'O','O'
        former_s,former_e = res[0]["start"], res[0]["end"]
        types, letters, starts, ends, words = [], [], [], [], []

        types.append(former_type)
        starts.append(former_s)
        letters.append(former_prefix)

        #### For each tag in level n
        for i in range(1,len(res)):
            #Get tag and prefix
            if res[i]["entity"] != 'O+O':
                type, letter =  res[i]["entity"].split('+')[j][2:], res[i]["entity"].split('+')[j][:1]
            else:
                type, letter = 'O', 'O'

            #If new tag or tag starting with B or b, create new entity
            if type != former_type or (type == former_type and letter in ['B','b']):
                former_type,former_prefix = type, letter
                types.append(type)
                letters.append(letter)
                starts.append(res[i]['start'])
                ends.append(res[i - 1]['end'])
        ends.append(res[len(res) - 1]['end'])

        for i in range(len(types)):
            words.append(entry[starts[i]:ends[i]])
        
        all_types.append(types)
        all_words.append(words)
        all_starts.append(starts)
        all_ends.append(ends)

    #Final res for one entry
    levels = {}
    for i in range(num_levels):
        df = pd.DataFrame(data={'label': all_types[i], 'start': all_starts[i], 'end': all_ends[i], 'span': all_words[i]})
        df.loc[df.label == '', 'label'] = 'O'
        df = df.sort_values(by='start',ignore_index=True, ascending=True)
        #print(df)
        levels[f"niv_{str(i+1)}"] = df.to_dict('records')
        #print(levels[f"niv_{str(i+1)}"])
    return levels, num_levels

################################### XML ##############################################
    
def xmlize_multilevel(levels,num_levels):
    """
    Params:
    levels (type : dict)
    num_levels (type : int)
    
    Return:
    Single entry with multi-levels xml tags
    """
    #Creation of xml root
    entry = etree.Element("entry")
    res = ''
    
    #Level one : initialisation
    for k in range(1,2):
        df = pd.DataFrame.from_dict(levels[f"niv_{k}"])
        # Create tags (all tokens are used)
        for i in range(len(df)):
            lab = df.iloc[i, 0]
            child = etree.SubElement(entry, str(lab))
            child.text = df['span'][i]
            child.get("start")
            child.set("start", str(df['start'][i]))
            child.get("end")
            child.set("end", str(df['end'][i]))
        etree.strip_tags(entry, 'O')

    #Other levels
    if len(pd.DataFrame.from_dict(levels[f"niv_2"])) != 0: #Uniquement s'il y a plus d'un niveau
        xml_tosearch = './/'
        for k in range(2,num_levels+1):
            df = pd.DataFrame.from_dict(levels[f"niv_{k}"])
            df = df.sort_values(by='start', ignore_index=True, ascending=False)

            #For each tag of level n
            for i in range(len(df)):
                lab, s, e, text = df.iloc[i, 0], int(df['start'][i]), int(df['end'][i]), df['span'][i]
                for p in entry.findall(xml_tosearch):
                    if s>=int(p.attrib['start']) and e <=int(p.attrib['end']) and lab != 'O':
                        n_s = s-int(p.attrib['start'])
                        n_e = n_s+len(text)
                        p.text = str(p.text[:n_s]) + '<' + lab + ' start="' + str(n_s) + '" end="' + str(n_e) + '">' + text +'</'+lab+'>' + p.text[n_e:]
            xml_tosearch += './/'

    #Clean res
    res = str(etree.tostring(entry, encoding=str, pretty_print=False))
    res = res.replace('&lt;', '<')
    res = res.replace('&gt;', '>')
    res = res.replace('<entry>','')
    res = res.replace('</entry>','')
    res = re.sub(' start="[0-9]+" end="[0-9]+"', '', res)
    return res

############################################ Other strategy ############################################

def group_sub_entities(entities):
    """
    Group together the adjacent tokens with the same entity predicted.

    Args:
        entities (:obj:`dict`): The entities predicted by the pipeline.
    """
    # Get the first entity in the entity group
    entity = entities[0]["entity"].split("-")[-1]
    scores = np.nanmean([entity["score"] for entity in entities])
    tokens = [entity["word"] for entity in entities]

    entity_group = {
        "entity_group": entity,
        "score": np.mean(scores),
        "word": tokenizer.convert_tokens_to_string(tokens),
        "start": entities[0]["start"],
        "end": entities[-1]["end"],
    }
    return entity_group

def get_tag(entity_name):
    if entity_name.startswith("B-"):
        bi = "B"
        tag = entity_name[2:]
    elif entity_name.startswith("I-"):
        bi = "I"
        tag = entity_name[2:]
    else:
        # It's not in B-, I- format
        # Default to I- for continuation.
        bi = "I"
        tag = entity_name
    return bi, tag

def group_entities(entities):
    """
    Find and group together the adjacent tokens with the same entity predicted.

    Args:
        entities (:obj:`dict`): The entities predicted by the pipeline.
    """

    entity_groups = []
    entity_group_disagg = []

    for entity in entities:
        if not entity_group_disagg:
            entity_group_disagg.append(entity)
            continue

        # If the current entity is similar and adjacent to the previous entity,
        # append it to the disaggregated entity group
        # The split is meant to account for the "B" and "I" prefixes
        # Shouldn't merge if both entities are B-type
        bi, tag = get_tag(entity["entity"])
        last_bi, last_tag = get_tag(entity_group_disagg[-1]["entity"])

        if tag == last_tag and bi != "B":
            # Modify subword type to be previous_type
            entity_group_disagg.append(entity)
        else:
            # If the current entity is different from the previous entity
            # aggregate the disaggregated entity group
            entity_groups.append(group_sub_entities(entity_group_disagg))
            entity_group_disagg = [entity]
    if entity_group_disagg:
        # it's the last entity, add it to the entity groups
        entity_groups.append(group_sub_entities(entity_group_disagg))

    return entity_groups

def get_by_level_entity_groups(res,level):
    level = level
    for d in res:
        d["entity"] = d["entity"].replace('_','-')
        d["entity"] = d["entity"].upper()
        if d["entity"] == 'O+O':
            d["entity"] = d["entity"].split('+')
        else:
            d["entity"] = d["entity"][2:].split('+')
        d["entity"] = d["entity"][level]
    res_ = group_entities(res)
    return res_