'''
Author: Zhenkun Shi
Date: 2020-06-05 05:10:25
LastEditors: Zhenkun Shi
LastEditTime: 2023-04-19 05:59:01
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
DIR_DICT = DATADIR +'dict/'


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


FILE_CASE_SHEWANELLA_FASTA=DATADIR+'shewanella.faa'


TRAIN_FEATURE = DATADIR+'train.feather'
TEST_FEATURE = DATADIR+'test.feather'
TRAIN_FASTA = DATADIR+'train.fasta'
TEST_FASTA = DATADIR+'test.fasta'

FILE_LATEST_SPROT = DATADIR + 'uniprot_sprot_latest.dat.gz'
FILE_LATEST_TREMBL = DATADIR + 'uniprot_trembl_latest.dat.gz'

FILE_LATEST_SPROT_FEATHER = DATADIR + 'uniprot/sprot_latest.feather'
FILE_LATEST_TREMBL_FEATHER = DATADIR + 'uniprot/trembl_latest.feather'


FILE_EC_LABEL_DICT = DATADIR + 'ec_label_dict.npy'
FILE_BLAST_TRAIN_DB = DATADIR + 'train_blast.dmnd' # blast比对数据库
FILE_BLAST_PRODUCTION_DB = DATADIR + 'uniprot_blast_db/production_blast.dmnd' # 生产环境比对数据库
FILE_BLAST_PRODUCTION_FASTA = DATADIR + 'production_blast.fasta' # 生产环境比对数据库
FILE_TRANSFER_DICT = DATADIR + 'ec_transfer_dict.npy'



ISENZYME_MODEL = MODELDIR+'/isenzyme.h5'
HOWMANY_MODEL = MODELDIR+'/howmany_enzyme.h5'
EC_MODEL = MODELDIR+'/ec.h5'


FILE_BLAST_RESULTS = RESULTSDIR + r'test_blast_res.tsv'
FILE_BLAST_ISENAYME_RESULTS = RESULTSDIR +r'isEnzyme_blast_results.tsv'
FILE_BLAST_EC_RESULTS = RESULTSDIR +r'ec_blast_results.tsv'


FILE_DEEPEC_RESULTS = RESULTSDIR + r'deepec/DeepEC_Result.txt'
FILE_ECPRED_RESULTS = RESULTSDIR + r'ecpred/ecpred.tsv'
FILE_CATFAM_RESULTS = RESULTSDIR + r'catfam_results.output'
FILE_PRIAM_RESULTS = RESULTSDIR + R'priam/PRIAM_20210819134344/ANNOTATION/sequenceECs.txt'

FILE_EVL_RESULTS = RESULTSDIR + r'evaluation_table.xlsx'

UPDATE_MODEL = True #强制模型更新标志
EMBEDDING_METHOD={  'one-hot':1, 
                    'unirep':2, 
                    'esm0':3, 
                    'esm32':4, 
                    'esm33':5
                }

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


DICT_LABEL_T1 = DIR_DICT+'dict_label_task1.h5'
DICT_LABEL_T2 = DIR_DICT+'dict_label_task2.h5'
DICT_LABEL_T3 = DIR_DICT+'dict_label_task3.h5'