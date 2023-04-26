import pandas as pd
import numpy as np
        
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


def getSubsetStats(set_):
    stats = []

    #init counters
    per = 0
    act = 0
    desc = 0
    titre = 0
    titreh = 0
    titrep = 0
    spat = 0
    loc = 0
    ft = 0
    cardinal = 0

    for i in range(len(set_)):
        per += set_[i][0].count('<PER>')
        act += set_[i][0].count('<ACT>')
        desc += set_[i][0].count('<DESC>')
        titre += set_[i][0].count('<TITRE>')
        titreh += set_[i][0].count('<TITREH>')
        titrep += set_[i][0].count('<TITREP>')
        spat += set_[i][0].count('<SPAT>')
        loc += set_[i][0].count('<LOC>')
        ft += set_[i][0].count('<FT>')
        cardinal += set_[i][0].count('<CARDINAL>')

    return {
        "PER":per,
        "ACT":act,
        "DESC":desc,
        "TITRE":titre,
        "TITREH":titreh,
        "TITREP":titrep,
        "SPAT":spat,
        "LOC":loc,
        "FT":ft,
        "CARDINAL":cardinal,
        }

def createStatsTab(train,dev,test):
    train_stats = getSubsetStats(train)
    dev_stats = getSubsetStats(dev)
    test_stats = getSubsetStats(test)

    types = pd.DataFrame({'Entity type': list(train_stats.keys())})
    train_counts = pd.DataFrame({'Train': list(train_stats.values())})
    dev_counts = pd.DataFrame({'Dev': list(dev_stats.values())})
    test_counts = pd.DataFrame({'Test': list(test_stats.values())})

    d = {'df':types, 'df1':train_counts, 'df2':dev_counts, 'df3':test_counts}

    first = True
    for key, value in d.items():
        if first:
            stats_df = value
            first = False
        else:
            stats_df = stats_df.merge(value, left_index=True, right_index=True)
            
    stats_df.set_index('Entity type', inplace=True)
    
    #Sum bu entity type
    stats_df['All'] = stats_df['Train'] + stats_df['Dev'] + stats_df['Test']
    
    #Sum by set
    sum = stats_df.sum()
    sum.name = 'All'
    stats_df = stats_df.append(sum.transpose())
    
    return stats_df