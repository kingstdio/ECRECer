import numpy as np
import pandas as pd
import time
import datetime
import sys
import os
from tqdm import tqdm
from functools import reduce
import joblib

sys.path.append("./tools/")
import funclib

import benchmark_train as btrain
import benchmark_test as btest
import config as cfg
import benchmark_evaluation as eva

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pandarallel import pandarallel #  import pandaralle
pandarallel.initialize() # init

if __name__ =='__main__':
    #read train test data
    train = pd.read_feather(cfg.DATADIR+'task3/train.feather')
    test = pd.read_feather(cfg.DATADIR+'task3/test.feather')
    print('train size: {0}\ntest size: {1}'.format(len(train), len(test)))

    train_set= funclib.split_ecdf_to_single_lines(train)
    test_set=funclib.split_ecdf_to_single_lines(test)

    #4. 加载EC号训练数据
    print('loading ec to label dict')
    if os.path.exists(cfg.FILE_EC_LABEL_DICT):
        dict_ec_label = np.load(cfg.FILE_EC_LABEL_DICT, allow_pickle=True).item()
    else:
        dict_ec_label = btrain.make_ec_label(train_label=train_set['ec_number'], test_label=test_set['ec_number'], file_save= cfg.FILE_EC_LABEL_DICT, force_model_update=cfg.UPDATE_MODEL)
        
    train_set['ec_label'] = train_set.ec_number.parallel_apply(lambda x: dict_ec_label.get(x))
    test_set['ec_label'] = test_set.ec_number.parallel_apply(lambda x: dict_ec_label.get(x))


    trainset = train_set.copy()
    testset = test_set.copy()
    encode_dict = dict(zip(set(trainset.ec_label),range(len(set(trainset.ec_label)))))
    trainset['ec_label_ecd']=trainset.ec_label.apply(lambda x: 0 if encode_dict.get(x)==None else encode_dict.get(x))
    testset['ec_label_ecd']=testset.ec_label.apply(lambda x: 0 if encode_dict.get(x)==None else encode_dict.get(x))


    MAX_SEQ_LENGTH = 1500 #定义序列最长的长度
    trainset.seq = trainset.seq.map(lambda x : x[0:MAX_SEQ_LENGTH].ljust(MAX_SEQ_LENGTH, 'X'))
    testset.seq = testset.seq.map(lambda x : x[0:MAX_SEQ_LENGTH].ljust(MAX_SEQ_LENGTH, 'X'))
    f_train = funclib.dna_onehot(trainset) #训练集编码
    f_test = funclib.dna_onehot(testset) #测试集编码


    # 计算指标
    X_train = np.array(f_train.iloc[:,2:])
    X_test = np.array(f_test.iloc[:,2:])
    Y_train = np.array(trainset.ec_label_ecd.astype('int'))
    Y_test = np.array(testset.ec_label_ecd.astype('int'))
    funclib.xgmain(X_train, Y_train, X_test, Y_test, type='multi')