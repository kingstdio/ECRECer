import re
import pandas as pd
import numpy as np
import joblib
import os
import benchmark_common as bcommon
import config as cfg
from Bio import SeqIO
import benchmark_test as btest
import argparse
import tools.funclib as funclib
import tools.embedding_esm as esmebd

parser = argparse.ArgumentParser()
parser.add_argument('-i', help='input file', default=cfg.DATADIR + 'test.fasta')
parser.add_argument('-o', help='output file', default=cfg.RESULTSDIR + 'ec_res.tsv')

def integrate_out_put(existing_table, blast_table, isEnzyme_pred_table, how_many_table, ec_table):
    result_set = pd.concat([existing_table, blast_table], axis=0)
    result_set = result_set.drop_duplicates(subset=['id'], keep='first').sort_values(by='seqlength')

    return result_set

def predict_function_counts(test_data):
    res=pd.DataFrame()
    res['id']=test_data.id
    model_s = joblib.load(cfg.MODELDIR+'/single_multi.model')
    model_m = joblib.load(cfg.MODELDIR+'/multi_many.model')
    pred_s=model_s.predict(np.array(test_data.iloc[:,1:]))
    pred_m=model_m.predict(np.array(test_data.iloc[:,1:]))
    res['pred_s']=1-pred_s
    res['pred_m']=pred_m+2

    return res





if __name__ =='__main__':
    args = parser.parse_args()
    # 1. 读入数据
    print('step 1: loading data') 
    input_df = funclib.load_fasta_to_table(args.i) # test fasta
    latest_sprot = pd.read_feather(cfg.FILE_LATEST_SPROT_FEATHER) #sprot db

    # 2. 查找数据
    print('step 2: find existing data')
    find_data =input_df.merge(latest_sprot, on='seq', how='left')
    exist_data= find_data[~find_data.name.isnull()].iloc[:,np.r_[0,2,1,12,7,9:12]].rename(columns={'id_x':'id','id_y':'id_uniprot'})
    noExist_data = find_data[find_data.name.isnull()]
    noExist_data.reset_index(drop=True, inplace=True)
    noExist_data = noExist_data.iloc[:,np.r_[0,2,1,12,7,9:12]].rename(columns={'id_x':'id','id_y':'id_uniprot'})

    # 3. EMBedding
    print('step 3: Embedding')
    rep0, rep32, rep33 = esmebd.get_rep_multi_sequence(sequences=noExist_data, model='esm1b_t33_650M_UR50S',seqthres=1022)

    # 4. 获取序列比对结果
    print('step 4: sequence alignment')
    if ~os.path.exists(cfg.FILE_BLAST_PRODUCTION_DB):
        funclib.table2fasta(latest_sprot, cfg.FILE_BLAST_PRODUCTION_FASTA)
        cmd = r'diamond makedb --in {0} -d {1}'.format(cfg.FILE_BLAST_PRODUCTION_FASTA, cfg.FILE_BLAST_PRODUCTION_DB)
        os.system(cmd)

    blast_res = funclib.getblast_usedb(db=cfg.FILE_BLAST_PRODUCTION_DB, test=noExist_data)
    blast_res = blast_res[['id', 'sseqid']].merge(latest_sprot, left_on='sseqid', right_on='id', how='left').iloc[:,np.r_[0,2:14]]
    blast_res = blast_res.iloc[:,np.r_[0,1,11,12,6,8:11]].rename(columns={'id_x':'id','id_y':'id_uniprot'})

    # 5. isEnzyme Prediction
    print('predict isEnzyme')
    model_isEnzyme = joblib.load(cfg.ISENZYME_MODEL)
    pred_isEnzyme = pd.DataFrame()
    pred_isEnzyme['id']=rep32.id
    pred_isEnzyme['isEnzyme_pred'] = model_isEnzyme.predict(rep32.iloc[:,1:])

    # 6. How many Prediction
    print('predict function counts')
    model_isEnzyme = joblib.load(cfg.ISENZYME_MODEL)
    # N. integrating

    print('integrate results')

    output_df = integrate_out_put(existing_table=exist_data,
                                  blast_table=blast_res,
                                  isEnzyme_pred_table = pd.NA, 
                                  how_many_table = pd.NA, 
                                  ec_table = pd.NA
                                )
                    
    output_df.to_csv( args.o, sep='\t')
    # # 3.获取酶-非酶预测结果
    # print('step 3. get isEnzyme results')
    # testX, testY = get_test_set(data=test)
    # isEnzyme_pred, isEnzyme_pred_prob = get_isEnzymeRes(querydata=testX, model_file=cfg.ISENZYME_MODEL)


    # # 4. 预测几功能酶预测结果
    # print('step 4. get howmany functions ')
    # howmany_Enzyme_pred, howmany_Enzyme_pred_prob = get_howmany_Enzyme(querydata=testX, model_file=cfg.HOWMANY_MODEL)

    # # 5.获取Slice预测结果
    # print('step 5. get EC prediction results')
    # # 5.1 准备slice所用文件
    # bcommon.prepare_slice_file( x_data=testX, 
    #                             y_data=testY['ec_number'], 
    #                             x_file=cfg.FILE_SLICE_TESTX,
    #                             y_file=cfg.FILE_SLICE_TESTY,
    #                             ec_label_dict=dict_ec_label
    #                         )
    # # 5.2 获得预测结果
    # # slice_pred_ec = get_slice_res(slice_query_file=cfg.FILE_SLICE_TESTX, model_path= cfg.MODELDIR, dict_ec_label=dict_ec_label, test_set=test,  res_file=cfg.FILE_SLICE_RESULTS)
    # slice_pred_ec = get_slice_res(slice_query_file=cfg.DATADIR+'slice_test_x_esm33.txt', model_path= cfg.MODELDIR+'/slice_esm33', dict_ec_label=dict_ec_label, test_set=test,  res_file=cfg.FILE_SLICE_RESULTS)
    
    # slice_pred_ec['isEnzyme_pred_xg'] = isEnzyme_pred
    # slice_pred_ec['functionCounts_pred_xg'] = howmany_Enzyme_pred
    # slice_pred_ec = slice_pred_ec.merge(blast_res, on='id', how='left')

    # # slice_pred_ec.to_csv(cfg.RESULTSDIR + 'singele_slice.tsv', sep='\t', index=None)
    # # 5.5 获取blast EC预测结果

    # # 6.将结果集成输出(slice_pred=slice_pred_ec, dict_ec_transfer=dict_ec_transfer)
    # slice_pred_ec = run_integrage(slice_pred=slice_pred_ec, dict_ec_transfer = dict_ec_transfer)    
    # slice_pred_ec.to_csv(cfg.FILE_INTE_RESULTS, sep='\t', index=None)

    # print('predict finished')