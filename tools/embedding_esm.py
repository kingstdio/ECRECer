from esm import model
import torch
import esm
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "../../")
import config as cfg


# region 将字符串拆分成固定长度
def cut_text(text,lenth):
    """[将字符串拆分成固定长度]

    Args:
        text ([string]): [input string]
        lenth ([int]): [sub_sequence length]

    Returns:
        [string list]: [string results list]
    """
    textArr = re.findall('.{'+str(lenth)+'}', text) 
    textArr.append(text[(len(textArr)*lenth):]) 
    return textArr 
#endregion

#region 对单个序列进行embedding
def get_rep_single_seq(seqid, sequence, model,batch_converter, seqthres=1022):
    """[对单个序列进行embedding]

    Args:
        seqid ([string]): [sequence name]]
        sequence ([sting]): [sequence]
        model ([model]): [ embedding model]]
        batch_converter ([object]): [description]
        seqthres (int, optional): [max sequence length]. Defaults to 1022.

    Returns:
        [type]: [description]
    """
    
    if len(sequence) < seqthres:
        data =[(seqid, sequence)]
    else:
        seqArray = cut_text(sequence, seqthres)
        data=[]
        for item in seqArray:
            data.append((seqid, item))
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    
    if torch.cuda.is_available():
        batch_tokens = batch_tokens.to(device="cuda", non_blocking=True)
        
    REP_LAYERS = [0, 32, 33]    
    MINI_SIZE = len(batch_labels)
    
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=REP_LAYERS, return_contacts=False)    
    

    representations = {layer: t.to(device="cpu") for layer, t in results["representations"].items()}
    result ={}
    result["label"] = batch_labels[0]

    for i in range(MINI_SIZE):
        if i ==0:
            result["mean_representations"] = {layer: t[i, 1 : len(batch_strs[0]) + 1].mean(0).clone() for layer, t in representations.items()}
        else:
            for index, layer in enumerate(REP_LAYERS):
                result["mean_representations"][layer] += {layer: t[i, 1 : len(batch_strs[0]) + 1].mean(0).clone() for layer, t in representations.items()}[layer]

    for index, layer in enumerate(REP_LAYERS):
        result["mean_representations"][layer] = result["mean_representations"][layer] /MINI_SIZE
    
    return result
#endregion

#region 对多个序列进行embedding
def get_rep_multi_sequence(sequences, model='esm_msa1b_t12_100M_UR50S', repr_layers=[0, 32, 33], seqthres=1022):
    """[对多个序列进行embedding]
    Args:
        sequences ([DataFrame]): [ sequence info]]
        seqthres (int, optional): [description]. Defaults to 1022.

    Returns:
        [DataFrame]: [final_rep0, final_rep32, final_rep33]
    """
    final_label_list = []
    final_rep0 =[]
    final_rep32 =[]
    final_rep33 =[]

    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    if torch.cuda.is_available():
            model = model.cuda()
            print("Transferred model to GPU")

    for i in tqdm(range(len(sequences))):
        apd = get_rep_single_seq(
            seqid = sequences.iloc[i].id, 
            sequence=sequences.iloc[i].seq, 
            model=model, 
            batch_converter=batch_converter, 
            seqthres=seqthres)

        final_label_list.append(np.array(apd['label']))
        final_rep0.append(np.array(apd['mean_representations'][0]))
        final_rep32.append(np.array(apd['mean_representations'][32]))
        final_rep33.append(np.array(apd['mean_representations'][33])) 

    final_rep0 = pd.DataFrame(final_rep0)
    final_rep32 = pd.DataFrame(final_rep32)
    final_rep33 = pd.DataFrame(final_rep33)
    final_rep0.insert(loc=0, column='id', value=np.array(final_label_list).flatten())
    final_rep32.insert(loc=0, column='id', value=np.array(final_label_list).flatten())
    final_rep33.insert(loc=0, column='id', value=np.array(final_label_list).flatten())

    col_name = ['id']+ ['f'+str(i) for i in range (1,final_rep0.shape[1])]
    final_rep0.columns = col_name
    final_rep32.columns = col_name
    final_rep33.columns = col_name

    return final_rep0, final_rep32, final_rep33
#endregion

if __name__ =='__main__':
    SEQTHRES = 1022
    RUNMODEL = {    'ESM-1b'    :'esm1b_t33_650M_UR50S', 
                'ESM-MSA-1b'    :'esm_msa1b_t12_100M_UR50S'
                }
    train = pd.read_feather(cfg.DATADIR+'train.feather').iloc[:,:6]
    test = pd.read_feather(cfg.DATADIR+'test.feather').iloc[:,:6]
    rep0, rep32, rep33 = get_rep_multi_sequence(sequences=train, model=RUNMODEL.get('ESM-MSA-1b'),seqthres=SEQTHRES)

    rep0.to_feather(cfg.DATADIR + 'train_rep0.feather')
    rep32.to_feather(cfg.DATADIR + 'train_rep32.feather')
    rep33.to_feather(cfg.DATADIR + 'train_rep33.feather')

    print('Embedding Success!')