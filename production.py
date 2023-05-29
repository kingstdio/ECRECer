import pandas as pd
import numpy as np
import joblib
import os,sys
import benchmark_common as bcommon
import config as cfg
import argparse
import tools.funclib as funclib
from tools.Attention import Attention
from keras.models import load_model
import tools.embedding_esm as esmebd
import time
from pandarallel import pandarallel #  import pandaralle


#region Integrate output
def integrate_out_put(existing_table, blast_table, dmlf_pred_table, mode='p', topnum=1):
    """[Integrate output]

    Args:
        existing_table ([DataFrame]): [db search results table]
        blast_table ([DataFrame]): [sequence alignment results table]
        isEnzyme_pred_table (DataFrame): [isEnzyme prediction results table]
        how_many_table ([DataFrame]): [function counts prediction results table]
        ec_table ([DataFrame]): [ec prediction table]

    Returns:
        [DataFrame]: [final results]
    """
    existing_table['res_type'] = 'db_match'
    blast_table['res_type']='blast_match'
    results_df = ec_table.merge(blast_table, on='id', how='left')

    function_df = how_many_table.copy()
    function_df = function_df.merge(isEnzyme_pred_table, on='id', how='left')
    function_df = function_df.merge(blast_table[['id', 'ec_number']], on='id', how='left')
    function_df['pred_function_counts']=function_df.parallel_apply(lambda x :integrate_enzyme_functioncounts(x.ec_number, x.isEnzyme_pred, x.pred_s, x.pred_m), axis=1)
    results_df = results_df.merge(function_df[['id','pred_function_counts']],on='id',how='left')

    results_df.loc[results_df[results_df.res_type.isnull()].index,'res_type']='dmlf_pred'
    results_df['pred_ec']=results_df.parallel_apply(lambda x: gather_ec_by_fc(x.iloc[3:23],x.ec_number, x.pred_function_counts), axis=1)
    results_df = results_df.iloc[:,np.r_[0,23,1,2,32,27:31]].rename(columns={'seq_x':'seq','seqlength_x':'seqlength'})

    

    if mode=='p':
        existing_table['pred_ec']=''
        result_set = pd.concat([existing_table, results_df], axis=0)
        result_set = result_set.drop_duplicates(subset=['id'], keep='first').sort_values(by='res_type')
        result_set['ec_number'] = result_set.apply(lambda x: x.pred_ec if str(x.ec_number)=='nan' else x.ec_number, axis=1)
        result_set.reset_index(drop=True, inplace=True)
        result_set = result_set.iloc[:,0:9]
        
        result_set['seqlength'] = result_set.seq.apply(lambda x: len(x))
        result_set['ec_number'] = result_set.ec_number.apply(lambda x: 'Non-Enzyme' if len(x)==1 else x)
        result_set = result_set.rename(columns={'ec_number':'ecrecer_pred_ec_number'})
        
        result_set = result_set[['id','ecrecer_pred_ec_number','seq','seqlength']]
    
    if mode =='r':
        result_set= results_df.merge(ec_table, on=['id'], how='left')
        result_set=result_set.iloc[:,np.r_[0:3,30,5:9, 4,10:30]]
        result_set = result_set.rename(columns=dict({'seq_x': 'seq','pred_ec': 'top0','top0_y': 'top1' },  **{'top'+str(i) : 'top'+str(i+1) for i in range(0, 20)}))
#         result_set = result_set.iloc[:,0:(8+topnum)]
#         result_set.loc[result_set[result_set.id.isin(existing_table.id)].index.values,'res_type']= 'db_match'
      
        result_set = result_set.iloc[:,np.r_[0, 2:4,8:(8+topnum)]]
    
    return result_set

#endregion

#region Predict Function Counts
def predict_function_counts(test_data):
    """[Predict Function Counts]

    Args:
        test_data ([DataFrame]): [DF contain protein ID and Seq]

    Returns:
        [DataFrame]: [col1:id, col2: single or multi; col3: multi counts]
    """
    res=pd.DataFrame()
    res['id']=test_data.id
    model_s = joblib.load(cfg.MODELDIR+'/single_multi.model')
    model_m = joblib.load(cfg.MODELDIR+'/multi_many.model')
    pred_s=model_s.predict(np.array(test_data.iloc[:,1:]))
    pred_m=model_m.predict(np.array(test_data.iloc[:,1:]))
    res['pred_s']=1-pred_s
    res['pred_m']=pred_m+2

    return res
#endregion

#region Integrate function counts by blast, single and multi
def integrate_enzyme_functioncounts(blast, isEnzyme, single, multi):
    """[Integrate function counts by blast, single and multi]

    Args:
        blast ([type]): [blast results]
        s ([type]): [single prediction]
        m ([type]): [multi prediction]

    Returns:
        [type]: [description]
    """
    if str(blast)!='nan':
        if str(blast)=='-':
            return 0
        else:
            return len(blast.split(','))
    if isEnzyme == 0:
        return 0
    if single ==1:
        return 1
    return multi
#endregion

#region format finnal ec by function counts
def gather_ec_by_fc(toplist, ec_blast ,counts):
    """[format finnal ec by function counts]

    Args:
        toplist ([list]): [top 20 predicted EC]
        ec_blast ([string]): [blast results]
        counts ([int]): [function counts]

    Returns:
        [string]: [comma sepreated ec string]
    """
    if counts==0:
        return '-'
    elif str(ec_blast)!='nan':
        return str(ec_blast)
    else:
        return ','.join(toplist[0:counts])
#endregion


#region run
def step_by_step_run(input_fasta, output_tsv, mode='p', topnum=1):
    """[run]
    Args:
        input_fasta ([string]): [input fasta file]
        output_tsv ([string]): [output tsv file]
    """
    start = time.process_time()
    if mode =='p':
        print('run in annoation mode')
    if mode =='r':
        print('run in recommendation mode')

    # 1. 读入数据
    print('step 1: loading data')
    input_df = funclib.load_fasta_to_table(input_fasta) # test fasta
    latest_sprot = pd.read_feather(cfg.FILE_LATEST_SPROT_FEATHER) #sprot db

    
    # 2. 查找数据
    print('step 2: find existing data')
    find_data = input_df.merge(latest_sprot, on='seq', how='left')
    find_data = latest_sprot[latest_sprot.seq.isin(input_df.seq)]
    exist_data = find_data.merge(input_df, on='seq', how='left').iloc[:,np.r_[8,0,1:8]].rename(columns={'id_x':'uniprot_id','id_y':'input_id'}).reset_index(drop=True)
    noExist_data = input_df[~input_df.seq.isin(find_data.seq)]

    # if len(noExist_data) == 0:
    #     exist_data=exist_data[['input_id','ec_number']].rename(columns={'ec_number':'ec_pred'})
    #     exist_data.to_csv(output_tsv, sep='\t', index=False)
    #     end = time.process_time()
    #     print('All done running time: %s Seconds'%(end-start))
    #     return
    
    # 3. EMBedding
    print('step 3: Embedding')
    rep0, rep32, rep33 = esmebd.get_rep_multi_sequence(sequences=input_df, model='esm1b_t33_650M_UR50S',seqthres=1022)
    if mode=='p':
        # 4. sequence alignment
        print('step 4: sequence alignment')
        if not os.path.exists(cfg.FILE_BLAST_PRODUCTION_DB):
            funclib.table2fasta(latest_sprot, cfg.FILE_BLAST_PRODUCTION_FASTA)
            cmd = r'diamond makedb --in {0} -d {1}'.format(cfg.FILE_BLAST_PRODUCTION_FASTA, cfg.FILE_BLAST_PRODUCTION_DB)
            os.system(cmd)


        blast_res = funclib.getblast_usedb(db=cfg.FILE_BLAST_PRODUCTION_DB, test=input_df)
        blast_res =blast_res[['id', 'sseqid']].merge(latest_sprot, left_on='sseqid', right_on='id', how='left').iloc[:,np.r_[0,2,3:10]].rename(columns={'id_x':'input_id','id_y':'uniprot_id'}).reset_index(drop=True)

        # 5. isEnzyme Prediction
        print('step 5: predict isEnzyme')
        pred_dmlf = pd.DataFrame(rep32.id.copy())
        model_isEnzyme = load_model(cfg.ISENZYME_MODEL,custom_objects={"Attention": Attention}, compile=False)
        predicted = model_isEnzyme.predict(np.array(rep32.iloc[:,1:]).reshape(rep32.shape[0],1,-1))
        encoder_t1=joblib.load(cfg.DICT_LABEL_T1)
        
        pred_dmlf['dmlf_isEnzyme']=(encoder_t1.inverse_transform(bcommon.props_to_onehot(predicted))).reshape(1,-1)[0]


        # 6. How many Prediction
        print('step 6: predict function counts')
        model_howmany = load_model(cfg.HOWMANY_MODEL,custom_objects={"Attention": Attention}, compile=False)
        predicted = model_howmany.predict(np.array(rep32.iloc[:,1:]).reshape(rep32.shape[0],1,-1))
        encoder_t2=joblib.load(cfg.DICT_LABEL_T2)
        pred_dmlf['dmlf_howmany']=(encoder_t2.inverse_transform(bcommon.props_to_onehot(predicted))).reshape(1,-1)[0]


        # 7. EC Prediction
        print('step 7: predict EC')
        model_ec = load_model(cfg.EC_MODEL,custom_objects={"Attention": Attention}, compile=False)
        predicted = model_ec.predict(np.array(rep32.iloc[:,1:]).reshape(rep32.shape[0],1,-1))
        encoder_t3=joblib.load(cfg.DICT_LABEL_T3)
        pred_dmlf['dmlf_ec']=[','.join(item) for item in (encoder_t3.inverse_transform(bcommon.props_to_onehot(predicted)))]


        print('step 8: integrate results')
        results = pred_dmlf.merge(exist_data, left_on='id',right_on='input_id', how='left') 
        results=results.fillna('#')
        results['ec_pred'] =results.apply(lambda x : x.ec_number if x.ec_number!='#' else ('-' if x.dmlf_isEnzyme==False else x.dmlf_ec) ,axis=1)
        output_df = results[['id', 'ec_pred']].rename(columns={'id':'id_input'})
    elif mode =='r':
        print('step 4: recommendation')
        label_model_ec = pd.read_feather(f'{cfg.MODELDIR}/task3_labels.feather').label_multi.to_list()
        model_ec = load_model('/home/shizhenkun/codebase/DMLF/model/task3_esm32_2022.h5',custom_objects={"Attention": Attention}, compile=False)
        predicted = model_ec.predict(np.array(rep32.iloc[:,1:]).reshape(rep32.shape[0],1,-1))
        output_df=pd.DataFrame()
        output_df['id']=input_df['id'].copy()
        output_df['ec_recomendations']=pd.DataFrame(predicted).apply(lambda x :sorted(dict(zip((label_model_ec), x)).items(),key = lambda x:x[1], reverse = True)[0:topnum], axis=1 ).values
    else:
        print(f'mode:{mode} not found')
        sys.exit()


    print('step 9: writting results') 

    output_df.to_csv(output_tsv, sep='\t', index=False)

    print(output_df)
    
    end = time.process_time()
    print('All done running time: %s Seconds'%(end-start))
#endregion


if __name__ =='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input file (fasta format)', type=str, default=cfg.DATADIR + 'sample_25.fasta')
    parser.add_argument('-o', help='output file (tsv table)', type=str, default=cfg.RESULTSDIR + 'ec_res20230206.tsv')
    parser.add_argument('-mode', help='compute mode. p: prediction, r: recommendation', type=str, default='r')
    parser.add_argument('-topk', help='recommendation records, min=1, max=20', type=int, default='5')

    pandarallel.initialize() #init
    args = parser.parse_args()
    input_file = args.i
    output_file = args.o
    compute_mode = args.mode
    topk = args.topk
    
    step_by_step_run(   input_fasta=input_file, 
                        output_tsv=output_file, 
                        mode=compute_mode, 
                        topnum=topk
                    )
    