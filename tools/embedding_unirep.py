import numpy as np
from numpy.core.defchararray import index
import pandas as pd
import os
from tqdm import tqdm
from jax_unirep import get_reps
import sys
sys.path.append("../")
import config as cfg


def getunirep(enzyme_noemzyme, step):
    unirep_res = []
    counter = 1

    enzyme_noemzyme = enzyme_noemzyme[['id','seq']]
    for i in tqdm(range(0, len(enzyme_noemzyme), step)):
        train_h_avg, train_h_final, train_c_final= get_reps(list(enzyme_noemzyme.seq[i:i+step]))
        checkpoint = np.hstack((np.array(enzyme_noemzyme[i:i+step]),train_h_avg))
        
        if counter == 1:
            unirep_res = np.array(checkpoint)
        else:
            unirep_res = np.concatenate((unirep_res,checkpoint))

        np.save(r'/tmp/task_unirep_'+str(counter)+'.tsv', checkpoint)

        if len(train_h_avg) != step:
            print('length not match')
        counter += 1

    unirep_res = pd.DataFrame(unirep_res)
    unirep_res=unirep_res.iloc[:, np.r_[0,2:1902]]
    col_name = ['id']+ ['f'+str(i) for i in range (1,train_h_avg.shape[1]+1)]
    unirep_res.columns = col_name
    return unirep_res


def load_file(file):
    return pd.read_csv(file,sep='\t')

def save_file(file, data):
    data = pd.DataFrame(data, columns=['id', 'seq'] + ['f'+str(i) for i in range(1, 1901) ])
    h5 = pd.HDFStore(file, 'w', complevel=4, complib='blosc')
    h5['data'] = data
    h5.close()

if __name__ == '__main__':
    train = pd.read_feather(cfg.DATADIR+'task1/train.feather')
    test = pd.read_feather(cfg.DATADIR+'task1/test.feather')

    outfile_train = cfg.DATADIR + 'train_unirep1.feather'
    outfile_test = cfg.DATADIR + 'test_unirep1.feather'

    STEP= 400
    res_train = getunirep(train[['id', 'seq']], STEP)
    res_test = getunirep(test[['id', 'seq']], STEP)

    save_file(outfile_train, res_train)
    save_file(outfile_test, res_test)

    print('unirep finish')