import numpy as np

def xmlize(text, annot):
    txt_l = [*text]
    annot_r = annot.copy()
    annot_r.reverse()
    for elem in annot_r:
        s_ix = elem["start"]
        e_ix = elem["end"]
        txt_l.insert(e_ix,f'</{elem["entity_group"]}>')
        if txt_l[s_ix] == ' ': # Patch entities starting with a whitespace
            s_ix += 1
        txt_l.insert(s_ix,f'<{elem["entity_group"]}>')
    xml = "".join(txt_l)
    xml = xml.replace('<O>','')
    xml = xml.replace('</O>','')
    return xml


############ Functions from the Hugging Face
#https://huggingface.co/transformers/v4.7.0/_modules/transformers/pipelines/token_classification.html

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