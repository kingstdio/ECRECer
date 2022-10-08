'''
Author: Zhenkun Shi
Date: 2022-10-08 06:15:36
LastEditors: Zhenkun Shi
LastEditTime: 2022-10-08 08:00:58
FilePath: /DMLF/tools/embdding_onehot.py
Description: 

Copyright (c) 2022 by tibd, All Rights Reserved. 
'''


import numpy as np
import pandas as pd
import os,sys,itertools
from datetime import datetime
from tqdm import tqdm
from pandarallel import pandarallel # 导入pandaralle
sys.path.append(os.path.dirname(os.path.realpath('__file__')))
sys.path.append('../')
import config as cfg

# amino acid 编码字典
prot_dict = dict(
                    A=0b000000000000000000000001, R=0b000000000000000000000010, N=0b000000000000000000000100,  
                    D=0b000000000000000000001000, C=0b000000000000000000010000, E=0b000000000000000000100000,  
                    Q=0b000000000000000001000000, G=0b000000000000000010000000, H=0b000000000000000100000000,  
                    O=0b000000000000001000000000, I=0b000000000000010000000000, L=0b000000000000100000000000, 
                    K=0b000000000001000000000000, M=0b000000000010000000000001, F=0b000000000100000000000000, 
                    P=0b000000001000000000000000, U=0b000000010000000000000000, S=0b000000100000000000000000, 
                    T=0b000001000000000000000000, W=0b000010000000000000000000, Y=0b000100000000000000000000, 
                    V=0b001000000000000000000000, B=0b010000000000000000000000, Z=0b100000000000000000000000, 
                    X=0b000000000000000000000000
                )

#region using one-hot to encode a amino acid sequence
def protein_sequence_one_hot(protein_seq, padding=False, padding_window=1500):
    """ using one-hot to encode a amino acid sequence

    Args:
        protein_seq (string): aminio acid sequence
        padding (bool, optional): padding to a fixed length. Defaults to False.
        padding_window (int, optional): padded sequence size. Defaults to 1500.

    Returns:
        _type_: _description_
    """
    res = [prot_dict.get(item) for item in protein_seq]
    if padding==True:
        if len(protein_seq)>=padding_window:
            res= res[:padding_window]
        else:
            res=np.pad(res, (0,(padding_window-len(protein_seq))), 'median')
    return list(res)
#endregion


#region encode amino acid sequences dataframe
def get_onehot(sequences, padding=True, padding_window=1500):
    """encode amino acid sequences dataframe

    Args:
        sequences (DataFrame): sequences dataframe cols name must contain ['id, 'seq']
        padding (bool, optional): if padding to a fixed length. Defaults to True.
        padding_window (int, optional): fixed padding size. Defaults to 1500.

    Returns:
        DataFrame: one-hot represented sequences DataFrame
    """
    sequences = sequences[['id','seq']].copy()
    sequences['onehot']=sequences.parallel_apply(lambda x: protein_sequence_one_hot(protein_seq=x.seq, padding=padding, padding_window=padding_window), axis=1)
    one_hot_rep = pd.DataFrame(np.array(list(itertools.chain(*sequences.onehot.values))).reshape(-1,(padding_window)), columns=['f'+str(i) for i in range (1,(padding_window+1))])
    one_hot_rep.insert(0,'id',value=sequences.id.values)

    return one_hot_rep
#endregion

if __name__ =='__main__':
    seqs = pd.DataFrame([['seq1', 'MTTSVIVAGARTPIGKLMGSLKDFSASELGAIAIKGALEKANVPAS'],
                         ['seq2', 'MAERAPRGEVAVMVAVQSALVDRPGMLATARGLSHFGEHCIGWLIL']
                        ], columns=['id', 'seq'])
    pandarallel.initialize()
    res = get_onehot(sequences=seqs, padding=True, padding_window=50)
    print(res)