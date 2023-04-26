from lxml import etree
import pandas as pd
import re
import numpy as np

def get_NER_tags(entry,res):
    """
    entry: a directory entry without
    res: list of nlp pipelines results of each layer 
    """
    #Number of entities levels
    num_levels = len(res)

    #Final res
    levels = {}
    
    for i in range(num_levels):
        #Pour chaque tag
        types, words, starts, ends = [],[],[],[]
        for j in range(len(res[i])):
            if entry[res[i][j]['start']:res[i][j]['end']][0] == ' ':
                types.append('O')
                words.append(entry[res[i][j]['start']:res[i][j]['end']][:1])
                starts.append(res[i][j]['start'])
                ends.append(res[i][j]['start']+1)
                
                types.append(res[i][j]['entity_group'])
                words.append(entry[res[i][j]['start']:res[i][j]['end']][1:])
                starts.append(res[i][j]['start']+1)
                ends.append(res[i][j]['end'])
            else:
                types.append(res[i][j]['entity_group'])
                words.append(entry[res[i][j]['start']:res[i][j]['end']])
                starts.append(res[i][j]['start'])
                ends.append(res[i][j]['end'])

        # df contenant les entités du niveau i
        df = pd.DataFrame(data={'label': types, 'start': starts, 'end': ends, 'span': words})
        df = df.sort_values(by='start',ignore_index=True, ascending=True)
        # Création de la liste de résultats associés à la clé niv_i dans le dictionnaire levels qui contient l'ensemble des résultats
        levels[f"niv_{str(i+1)}"] = df.to_dict('records')
        
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
        #etree.strip_tags(entry, 'O')

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
    res = res.replace('</O>','')
    res = res.replace('<O>','')
    return res


############################################ Other strategy ############################################

def group_sub_entities(entities,tokenizer):
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
        bi = "B"
        tag = entity_name
    return bi, tag

def group_entities(entities,tokenizer):
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
            entity_groups.append(group_sub_entities(entity_group_disagg,tokenizer))
            entity_group_disagg = [entity]
    if entity_group_disagg:
        # it's the last entity, add it to the entity groups
        entity_groups.append(group_sub_entities(entity_group_disagg,tokenizer))

    return entity_groups