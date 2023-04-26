from lxml import etree
import pandas as pd
import re

def get_NER_tags(entry,res):
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
        types, prefixs, starts, ends, words = [], [], [], [], []

        types.append(former_type)
        starts.append(former_s)
        prefixs.append(former_prefix)

        #### For each tag in level n
        ###
        for i in range(1,len(res)):
            #Get tag and prefix
            if res[i]["entity"] != 'O+O':
                type, prefix =  res[i]["entity"].split('+')[j][2:], res[i]["entity"].split('+')[j][:1]
            else:
                type, prefix = 'O', 'O'

            #If new tag or tag starting with B or b, create new entity
            if type != former_type:
                    former_type, former_prefix = type, prefix
                    types.append(type)
                    prefixs.append(prefix)
                    starts.append(res[i]['start'])
                    ends.append(res[i - 1]['end'])
            elif type == former_type and prefix == 'b' and former_prefix == 'i':
                    former_type, former_prefix = type, prefix
                    types.append(type)
                    prefixs.append(prefix)
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
        print(df)
        levels[f"niv_{str(i+1)}"] = df.to_dict('records')
        #print(levels[f"niv_{str(i+1)}"])
    return levels, num_levels

## Comment gérer les scores égaux ?

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