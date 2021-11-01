# 1. 定义数据目录
DATADIR = r'''/home/shizhenkun/codebase/DMLF/data/'''
RESULTSDIR = r'''/home/shizhenkun/codebase/BioUniprot/data/benchmark/results/'''
MODELDIR = r'''/home/shizhenkun/codebase/DMLF/model'''
TEMPDIR =r'''/home/shizhenkun/codebase/DMLF/tmp/'''


TRAIN_FEATURE = DATADIR+'train.feather'
TEST_FEATURE = DATADIR+'test.feather'
TRAIN_FASTA = DATADIR+'train.fasta'
TEST_FASTA = DATADIR+'test.fasta'
FILE_SLICE_TRAINX = DATADIR + 'slice_train_x.txt'
FILE_SLICE_TRAINY  = DATADIR + 'slice_train_y.txt'
FILE_SLICE_TESTX = DATADIR + 'slice_test_x.txt'
FILE_SLICE_TESTY  = DATADIR + 'slice_test_y.txt'
FILE_EC_LABEL_DICT = DATADIR + 'ec_label_dict.npy'
FILE_BLAST_TRAIN_DB = DATADIR + 'train_blast.dmnd' # blast比对数据库
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

UPDATE_MODEL = True #强制模型更新标志
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
TRAIN_ISENZYME_LEARNING_STEPS = 100             #是否是酶学习次数
TRAIN_HOWMANY_ENZYME_LEARNING_STEPS = 100        #几功能酶学习次数
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
