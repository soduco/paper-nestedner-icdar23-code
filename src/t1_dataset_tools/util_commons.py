from operator import itemgetter

def unwrap(list_of_tuples2):
    return tuple(zip(*list_of_tuples2))

def file_name(*parts, separator="_"):
    parts_str = [str(p) for p in parts]
    return separator.join(filter(None,parts_str))

def getAnn(path):
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        ann = []
        for e in lines:
            if e != '\n':
                labels = e.split('	')
                m = labels[1].split(' ')
                res = [m[0], int(m[1]), int(m[2]),int(m[2])-int(m[1])]
                ann.append(res)
        tmp = sorted(ann, key=itemgetter(1, 3))

        final_ann = []
        for t in tmp:
            f = [t[0],[t[1],t[2]]]
            final_ann.append(f)
        return final_ann
    
    
def orderLabelsbyEntsLevel(ents_list):
    order_list = []
    if len(ents_list) == 2:
        if "TITREH" in "".join(ents_list):
            n1 = [ent for ent in ents_list if "PER" in ent][0]
            n2 = [ent for ent in ents_list if "TITREH" in ent][0]
            n2 = n2.replace('-H','H')
            order_list = [n1,n2]
            
        elif "DESC" in "".join(ents_list) and "ACT" in "".join(ents_list):
            n1 = [ent for ent in ents_list if "DESC" in ent][0]
            n2 = [ent for ent in ents_list if "ACT" in ent][0]
            order_list = [n1,n2]
            
        elif "DESC" in "".join(ents_list) and "TITREP" in "".join(ents_list):
            n1 = [ent for ent in ents_list if "DESC" in ent][0]
            n2 = [ent for ent in ents_list if "TITREP" in ent][0]
            n2 = n2.replace('-P','P')
            order_list = [n1,n2]
            
        elif "SPAT" in "".join(ents_list) and "LOC" in "".join(ents_list):
            n1 = [ent for ent in ents_list if "SPAT" in ent][0]
            n2 = [ent for ent in ents_list if "LOC" in ent][0]
            order_list = [n1,n2]
            
        elif "SPAT" in "".join(ents_list) and "CARDINAL" in "".join(ents_list):
            n1 = [ent for ent in ents_list if "SPAT" in ent][0]
            n2 = [ent for ent in ents_list if "CARDINAL" in ent][0]
            order_list = [n1,n2]
            
        elif "SPAT" in "".join(ents_list) and "FT" in "".join(ents_list):
            n1 = [ent for ent in ents_list if "SPAT" in ent][0]
            n2 = [ent for ent in ents_list if "FT" in ent][0]
            order_list = [n1,n2]
            
    elif len(ents_list) == 1:
        if "PER" in "".join(ents_list):
            n1 = [ent for ent in ents_list if "PER" in ent][0]
            order_list = [n1,'O']
        elif 'ACT' in "".join(ents_list):
            n1 = [ent for ent in ents_list if "ACT" in ent][0]
            order_list = [n1,'O']
        elif 'DESC' in "".join(ents_list):
            n1 = [ent for ent in ents_list if "DESC" in ent][0]
            order_list = [n1,'O']
        elif 'SPAT' in "".join(ents_list):
            n1 = [ent for ent in ents_list if "SPAT" in ent][0]
            order_list = [n1,'O']
        elif 'TITRE' in "".join(ents_list):
            n1 = [ent for ent in ents_list if "TITRE" in ent][0]
            order_list = [n1,'O']
    else:
        order_list = ['O','O']
    return order_list