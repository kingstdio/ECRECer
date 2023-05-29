'''
Author: Zhenkun Shi
Date: 2022-10-07 14:04:25
LastEditors: Zhenkun Shi
LastEditTime: 2023-04-04 18:56:49
FilePath: /DMLF/tools/minitools.py
Description: 

Copyright (c) 2022 by tibd, All Rights Reserved. 
'''
import numpy as np
import pandas as pd


def convert_DF_dateTime(inputdf):
    """[Covert unisprot csv records datatime]

    Args:
        inputdf ([DataFrame]): [input dataFrame]

    Returns:
        [DataFrame]: [converted DataFrame]
    """
    inputdf.date_integraged = pd.to_datetime(inputdf['date_integraged'])
    inputdf.date_sequence_update = pd.to_datetime(inputdf['date_sequence_update'])
    inputdf.date_annotation_update = pd.to_datetime(inputdf['date_annotation_update'])
    inputdf = inputdf.sort_values(by='date_integraged', ascending=True)
    inputdf.reset_index(drop=True, inplace=True)
    return inputdf


# add missing '-' for ec number
def refill_ec(ec):   
    if ec == '-':
        return ec
    levelArray = ec.split('.')
    if  levelArray[3]=='':
        levelArray[3] ='-'
    ec = '.'.join(levelArray)
    return ec

def specific_ecs(ecstr):
    if '-' not in ecstr or len(ecstr)<4:
        return ecstr
    ecs = ecstr.split(',')
    if len(ecs)==1:
        return ecstr
    
    reslist=[]
    
    for ec in ecs:
        recs = ecs.copy()
        recs.remove(ec)
        ecarray = np.array([x.split('.') for x in recs])
        
        if '-' not in ec:
            reslist +=[ec]
            continue
        linearray= ec.split('.')
        if linearray[1] == '-':
            #l1 in l1s and l2 not empty
            if (linearray[0] in  ecarray[:,0]) and (len(set(ecarray[:,0]) - set({'-'}))>0):
                continue
        if linearray[2] == '-':
            # l1, l2 in l1s l2s, l3 not empty
            if (linearray[0] in  ecarray[:,0]) and (linearray[1] in  ecarray[:,1]) and (len(set(ecarray[:,2]) - set({'-'}))>0):
                continue
        if linearray[3] == '-':
            # l1, l2, l3 in l1s l2s l3s, l4 not empty
            if (linearray[0] in  ecarray[:,0]) and (linearray[1] in  ecarray[:,1]) and (linearray[2] in  ecarray[:,2]) and (len(set(ecarray[:,3]) - set({'-'}))>0):
                continue
                
        reslist +=[ec]
    return ','.join(reslist)

#format ec
def format_ec(ecstr):
    ecArray= ecstr.split(',')
    ecArray=[x.strip() for x in ecArray] #strip blank
    ecArray=[refill_ec(x) for x in ecArray] #format ec to full
    ecArray = list(set(ecArray)) # remove duplicates
    
    return ','.join(ecArray)