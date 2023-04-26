import glob
import re
import numpy as np

def preds_files_list(path_to_preds_files):
    """
    Params :
    - path_to_preds_file : root to prediction files folder
    
    Return predictions made on test dataset list
    """
    files = glob.glob(path_to_preds_files)
    t = []
    for file in files:
        trainsize = re.findall("test_(.*).txt", file)
        run = re.findall("run_([0-9]+)",file)
        t.append([file,int(trainsize[0]),int(run[0])])
    t = sorted(t, key=lambda x: (x[2],x[1])) #Sort path by trainset size
    return t
        
def unique(lists):
    """
    Params:
        lists : list of n lists of values
    Return a list of unique values
    """
    tuple_ = ()
    for i in range(len(lists)):
        x = np.array(lists[i])
        tuple_ = tuple_ + (x,)
    tab = np.concatenate(tuple_)
    return list(np.unique(tab))


def get_labels_names(labels,LABELS_ID):
    """
    Params : 
    - labels:list : liste de labels au format I-0+0 
    Retourne liste de labels au format I-O+O
    """
    labels_names = []
    key_list = list(LABELS_ID.keys())
    for elem in labels:
        labels_part = elem.split('+')
        lab1 = labels_part[0][2:]
        lab1, lab2 = int(lab1), int(labels_part[1])
        lab1, lab2 = key_list[lab1], key_list[lab2]
        text_label = str(lab1) + '+' + str(lab2) #Joint-label
        labels_names.append(text_label)
    return labels_names