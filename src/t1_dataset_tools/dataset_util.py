import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from datasets.dataset_dict import DatasetDict
import config
from datasets import load_from_disk

def unwrap(list_of_tuples2):
    #Itérateur
    #retourne les résultats sous la forme de tuples
    return tuple(zip(*list_of_tuples2))

def file_name(*parts, separator="_"):
    parts_str = [str(p) for p in parts]
    return separator.join(filter(None,parts_str))

def train_dev_test_split(gold):
    """Splits the gold dataset into two subsets where entries from the same directory is garanteed
    to not be in the both sets.
    Subset 1 is the test dataset with approx. ~20% of all entries.
    Subset 2 is splitted into a train set (~70%) and a validation set (~8%).
    For the second split we use stratified sampling based on directories names so the initial proportion
    of entries from each directory is preserved accros subsets.
    """

    # Use GroupKFold to pick a few directories for testing .
    # In the current state of the gold dataset (08/01/2022),
    # entries from 3 dirctories will be selected:
    #   - Bottin1_1820 (267 entries)
    #   - Didot_1851a (1266 entries)
    #   - Duverneuil_et_La_Tynna_1806 (186 entries)
    
    _, groups = unwrap(gold)
    index_tmp, index_test = list(GroupKFold(n_splits=5).split(gold, groups=groups))[0]
    subset_tmp, test = gold[index_tmp], gold[index_test]

    # Split subset_tmp into train (~90%) and dev (~10%) stratified on directories names
    _, groups = unwrap(subset_tmp)
    train_dev = train_test_split(
        subset_tmp, train_size=0.9, shuffle=True, random_state=config.SPLIT_SEED, stratify=groups
    )
    train = train_dev[0]
    dev = train_dev[1]
    return train, dev, test