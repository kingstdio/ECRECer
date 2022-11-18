'''
Author: Zhenkun Shi
Date: 2020-06-05 05:10:25
LastEditors: Zhenkun Shi
LastEditTime: 2022-11-17 13:23:12
FilePath: /DMLF/config.py
Description: 

Copyright (c) 2022 by tibd, All Rights Reserved. 
'''

import os


# 1. 定义数据目录
ROOTDIR= f'{os.getcwd()}/'
DATADIR = ROOTDIR +'data/'
RESULTSDIR = ROOTDIR +'results/'
MODELDIR = ROOTDIR +'model'
TEMPDIR =ROOTDIR +'tmp/'
DIR_UNIPROT = DATADIR + 'uniprot/'
DIR_DATASETS = DATADIR +'datasets/'
DIR_FEATURES = DATADIR + 'featureBank/'


#2.URL
URL_SPROT_SNAP201802 = f'https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2018_02/knowledgebase/uniprot_sprot-only2018_02.tar.gz'
URL_SPROT_SNAP201902 = f'https://ftp.uniprot.org/pub/databases/uniprot/previous_major_releases/release-2019_02/knowledgebase/uniprot_sprot-only2019_02.tar.gz'
URL_SPROT_SNAP202006 = f'https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2020_06/knowledgebase/uniprot_sprot-only2020_06.tar.gz'
URL_SPROT_SNAP202102 = f'https://ftp.uniprot.org/pub/databases/uniprot/previous_major_releases/release-2021_02/knowledgebase/uniprot_sprot-only2021_02.tar.gz'
URL_SPROT_SNAP202202 = f'https://ftp.uniprot.org/pub/databases/uniprot/previous_releases/release-2022_02/knowledgebase/uniprot_sprot-only2022_02.tar.gz'
URL_SPROT_LATEST=f'https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.dat.gz'

#3.FILES
FILE_SPROT_SNAP201802=DIR_UNIPROT+'uniprot_sprot-only2018_02.tar.gz'
FILE_SPROT_SNAP201902=DIR_UNIPROT+'uniprot_sprot-only2019_02.tar.gz'
FILE_SPROT_SNAP202006=DIR_UNIPROT+'uniprot_sprot-only2020_06.tar.gz'
FILE_SPROT_SNAP202102=DIR_UNIPROT+'uniprot_sprot-only2021_02.tar.gz'
FILE_SPROT_SNAP202202=DIR_UNIPROT+'uniprot_sprot-only2022_02.tar.gz'

FILE_SPROT_LATEST=DIR_UNIPROT+'uniprot_sprot_leatest.dat.gz'

FILE_FEATURE_UNIREP = DIR_FEATURES + 'embd_unirep.feather'
FILE_FEATURE_ESM0 = DIR_FEATURES + 'embd_esm0.feather'
FILE_FEATURE_ESM32 = DIR_FEATURES + 'embd_esm32.feather'
FILE_FEATURE_ESM33 = DIR_FEATURES + 'embd_esm33.feather'
FILE_FEATURE_ONEHOT = DIR_FEATURES + 'embd_onehot.feather'


FILE_TASK1_TRAIN = DIR_DATASETS + 'task1/train.feather'
FILE_TASK1_TEST_2019 = DIR_DATASETS + 'task1/test_2019.feather'
FILE_TASK1_TEST_2020 = DIR_DATASETS + 'task1/test_2020.feather'
FILE_TASK1_TEST_2021 = DIR_DATASETS + 'task1/test_2021.feather'
FILE_TASK1_TEST_2022 = DIR_DATASETS + 'task1/test_2022.feather'

FILE_TASK2_TRAIN = DIR_DATASETS + 'task2/train.feather'
FILE_TASK2_TEST_2019 = DIR_DATASETS + 'task2/test_2019.feather'
FILE_TASK2_TEST_2020 = DIR_DATASETS + 'task2/test_2020.feather'
FILE_TASK2_TEST_2021 = DIR_DATASETS + 'task2/test_2021.feather'
FILE_TASK2_TEST_2022 = DIR_DATASETS + 'task2/test_2022.feather'


FILE_TASK3_TRAIN = DIR_DATASETS + 'task3/train.feather'
FILE_TASK3_TEST_2019 = DIR_DATASETS + 'task3/test_2019.feather'
FILE_TASK3_TEST_2020 = DIR_DATASETS + 'task3/test_2020.feather'
FILE_TASK3_TEST_2021 = DIR_DATASETS + 'task3/test_2021.feather'
FILE_TASK3_TEST_2022 = DIR_DATASETS + 'task3/test_2022.feather'

FILE_TASK1_TRAIN_FASTA = DIR_DATASETS +'task1/train.fasta'
FILE_TASK1_TEST_2019_FASTA = DIR_DATASETS +'task1/test_2019.fasta'
FILE_TASK1_TEST_2020_FASTA = DIR_DATASETS +'task1/test_2020.fasta'
FILE_TASK1_TEST_2021_FASTA = DIR_DATASETS +'task1/test_2021.fasta'
FILE_TASK1_TEST_2022_FASTA = DIR_DATASETS +'task1/test_2022.fasta'

FILE_TASK2_TRAIN_FASTA = DIR_DATASETS +'task2/train.fasta'
FILE_TASK2_TEST_2019_FASTA = DIR_DATASETS +'task2/test_2019.fasta'
FILE_TASK2_TEST_2020_FASTA = DIR_DATASETS +'task2/test_2020.fasta'
FILE_TASK2_TEST_2021_FASTA = DIR_DATASETS +'task2/test_2021.fasta'
FILE_TASK2_TEST_2022_FASTA = DIR_DATASETS +'task2/test_2022.fasta'

FILE_TASK3_TRAIN_FASTA = DIR_DATASETS +'task3/train.fasta'
FILE_TASK3_TEST_2019_FASTA = DIR_DATASETS +'task3/test_2019.fasta'
FILE_TASK3_TEST_2020_FASTA = DIR_DATASETS +'task3/test_2020.fasta'
FILE_TASK3_TEST_2021_FASTA = DIR_DATASETS +'task3/test_2021.fasta'
FILE_TASK3_TEST_2022_FASTA = DIR_DATASETS +'task3/test_2022.fasta'




TRAIN_FEATURE = DATADIR+'train.feather'
TEST_FEATURE = DATADIR+'test.feather'
TRAIN_FASTA = DATADIR+'train.fasta'
TEST_FASTA = DATADIR+'test.fasta'

FILE_LATEST_SPROT = DATADIR + 'uniprot_sprot_latest.dat.gz'
FILE_LATEST_TREMBL = DATADIR + 'uniprot_trembl_latest.dat.gz'

FILE_LATEST_SPROT_FEATHER = DATADIR + 'latest_sprot.feather'
FILE_LATEST_TREMBL_FEATHER = DATADIR + 'latest_trembl.feather'

FILE_SLICE_TRAINX = DATADIR + 'slice_train_x.txt'
FILE_SLICE_TRAINY  = DATADIR + 'slice_train_y.txt'
FILE_SLICE_TESTX = DATADIR + 'slice_test_x.txt'
FILE_SLICE_TESTY  = DATADIR + 'slice_test_y.txt'
FILE_EC_LABEL_DICT = DATADIR + 'ec_label_dict.npy'
FILE_BLAST_TRAIN_DB = DATADIR + 'train_blast.dmnd' # blast比对数据库
FILE_BLAST_PRODUCTION_DB = DATADIR + 'production_blast.dmnd' # 生产环境比对数据库
FILE_BLAST_PRODUCTION_FASTA = DATADIR + 'production_blast.fasta' # 生产环境比对数据库
FILE_TRANSFER_DICT = DATADIR + 'ec_transfer_dict.npy'


ISENZYME_MODEL = MODELDIR+'/isenzyme.model'
HOWMANY_MODEL = MODELDIR+'/howmany_enzyme.model'


FILE_BLAST_RESULTS = RESULTSDIR + r'test_blast_res.tsv'
FILE_BLAST_ISENAYME_RESULTS = RESULTSDIR +r'isEnzyme_blast_results.tsv'
FILE_BLAST_EC_RESULTS = RESULTSDIR +r'ec_blast_results.tsv'
FILE_SLICE_ISENZYME_RESULTS = RESULTSDIR + 'isEnzyme_slice_results.tsv'

FILE_SLICE_RESULTS = RESULTSDIR + 'slice_results.txt'
FILE_INTE_RESULTS  =   RESULTSDIR+'slice_pred.tsv'
FILE_DEEPEC_RESULTS = RESULTSDIR + r'deepec/DeepEC_Result.txt'
FILE_ECPRED_RESULTS = RESULTSDIR + r'ecpred/ecpred.tsv'
FILE_CATFAM_RESULTS = RESULTSDIR + r'catfam_results.output'
FILE_PRIAM_RESULTS = RESULTSDIR + R'priam/PRIAM_20210819134344/ANNOTATION/sequenceECs.txt'

FILE_EVL_RESULTS = RESULTSDIR + r'evaluation_table.xlsx'

UPDATE_MODEL = False #强制模型更新标志
EMBEDDING_METHOD={  'one-hot':1, 
                    'unirep':2, 
                    'esm0':3, 
                    'esm32':4, 
                    'esm33':5
                }



# 训练参数
SAMPLING_BIT = 6 #采样精度


FEATURE_SET ={  'one-hot':1,
                'uni-rep':2,
                'esm':3
            }

# SLICE
FEATURE_NUM = 1900
TRAIN_USE_ONLY_SINGLE_FUNCTION = False           #只用单功能酶
TRAIN_USE_SPCIFIC_EC_LEVEL = 2                  #训练用ec级别大于n位的
TRAIN_USE_ONLY_ENZYME = True                    #只用酶数据进行训练
TRAIN_BLAST_IDENTITY_THRES = 40                  #比对结果identity阈值

# XGBoost
TRAIN_ISENZYME_LEARNING_STEPS = 1000             #是否是酶学习次数
TRAIN_HOWMANY_ENZYME_LEARNING_STEPS = 300        #几功能酶学习次数
VALIDATION_RATE = 0.3 #模型训练时验证集的比例    


#
BLAST_TABLE_HEAD = ['id', 
                    'sseqid', 
                    'pident', 
                    'length', 
                    'mismatch', 
                    'gapopen', 
                    'qstart', 
                    'qend', 
                    'sstart', 
                    'send',
                    'evalue', 
                    'bitscore'
                    ]
