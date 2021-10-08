from operator import index
from random import sample
import pandas as pd
import numpy as np
from sklearn import metrics
from tools import ucTools
from pandas._config.config import reset_option
import benchmark_common as bcommon
import benchmark_config as cfg


#region 加载各种算法的预测结果，并拼合成一个大表格
def load_res_data(file_slice, file_blast, file_deepec, file_ecpred,file_catfam, file_priam, train, test):
    """[加载各种算法的预测结果，并拼合成一个大表格]

    Args:
        file_slice ([string]]): [slice 集成预测结果文件]
        file_blast ([string]): [blast 序列比对结果文件]
        file_deepec ([string]): [deepec 预测结果文件]
        file_ecpred ([string]): [ecpred 预测结果文件]
        train ([DataFrame]): [训练集]
        test ([DataFrame]): [测试集]

    Returns:
        [DataFrame]: [拼合完成的大表格]
    """


    # Slice + groundtruth
    ground_truth = test.iloc[:, np.r_[0,1,2,3]]
    res_slice = pd.read_csv(file_slice, sep='\t')
    
    big_res = ground_truth.merge(res_slice, on='id', how='left')

    #Blast
    res_blast = pd.read_csv(file_blast, sep='\t',  names =cfg.BLAST_TABLE_HEAD)
    res_blast = bcommon.blast_add_label(blast_df=res_blast, trainset=train)
    big_res = big_res.merge(res_blast, on='id', how='left') 


    # DeepEC
    res_deepec = pd.read_csv(file_deepec, sep='\t',names=['id', 'ec_number'], header=0 )
    res_deepec.ec_number=res_deepec.apply(lambda x: x['ec_number'].replace('EC:',''), axis=1)
    res_deepec.columns = ['id','ec_deepec']
    res_deepec['isemzyme_deepec']=res_deepec.ec_deepec.apply(lambda x: True if str(x)!='nan' else False)
    res_deepec['functionCounts_deepec'] = res_deepec.ec_deepec.apply(lambda x :len(str(x).split(',')))
    big_res = big_res.merge(res_deepec, on='id', how='left').drop_duplicates(subset='id')

    # ECpred
    res_ecpred = pd.read_csv(file_ecpred, sep='\t', header=0)
    res_ecpred['isemzyme_ecpred'] = ''
    with pd.option_context('mode.chained_assignment', None):
        res_ecpred.isemzyme_ecpred[res_ecpred['EC Number']=='non Enzyme'] = False
        res_ecpred.isemzyme_ecpred[res_ecpred['EC Number']!='non Enzyme'] = True
        
    res_ecpred.columns = ['id','ec_ecpred', 'conf', 'isemzyme_ecpred']
    res_ecpred = res_ecpred.iloc[:,np.r_[0,1,3]]
    res_ecpred['functionCounts_ecpred'] = res_ecpred.ec_ecpred.apply(lambda x :len(str(x).split(',')))
    big_res = big_res.merge(res_ecpred, on='id', how='left').drop_duplicates(subset='id')

    # CATFAM
    res_catfam = pd.read_csv(file_catfam, sep='\t', names=['id', 'ec_catfam'])
    res_catfam['isenzyme_catfam']=res_catfam.ec_catfam.apply(lambda x: True if str(x)!='nan' else False)
    res_catfam['functionCounts_catfam'] = res_catfam.ec_catfam.apply(lambda x :len(str(x).split(',')))
    big_res = big_res.merge(res_catfam, on='id', how='left')

    #PRIAM
    res_priam = load_praim_res(resfile=file_priam)
    big_res = big_res.merge(res_priam, on='id', how='left')
    big_res['isenzyme_priam'] = big_res.ec_priam.apply(lambda x: True if str(x)!='nan' else False)
    big_res['functionCounts_priam'] = big_res.ec_priam.apply(lambda x :len(str(x).split(',')))
    
    big_res=big_res.sort_values(by=['isemzyme', 'id'], ascending=False)
    big_res = big_res.drop_duplicates(subset='id')
    big_res.reset_index(drop=True, inplace=True)
    return big_res
#endregion

#region
def load_praim_res(resfile):
    """[加载PRIAM的预测结果]
    Args:
        resfile ([string]): [结果文件]
    Returns:
        [DataFrame]: [结果]
    """
    f = open(cfg.FILE_PRIAM_RESULTS)
    line = f.readline()
    counter =0
    reslist=[]
    lstr =''
    subec=[]
    while line:
        if '>' in line:
            if counter !=0:
                reslist +=[[lstr, ', '.join(subec)]]
                subec=[]
            lstr = line.replace('>', '').replace('\n', '')
        elif line.strip()!='':
            ecarray = line.split('\t')
            subec += [(ecarray[0].replace('#', '').replace('\n', '').replace(' ', '') )]

        line = f.readline()
        counter +=1
    f.close()
    res_priam=pd.DataFrame(reslist, columns=['id', 'ec_priam'])
    return res_priam
#endregion

def integrate_reslults(big_table):
    # 拼合多个标签
    big_table['ec_islice']=big_table.apply(lambda x : ', '.join(x.iloc[4:(x.pred_functionCounts+4)].values.astype('str')), axis=1)

    #给非酶赋-
    with pd.option_context('mode.chained_assignment', None):
        big_table.ec_islice[big_table.ec_islice=='nan']='-'
        big_table.ec_islice[big_table.ec_islice=='']='-'
        big_table['isemzyme_deepec']=big_table.ec_deepec.apply(lambda x : 0 if str(x)=='nan' else 1) 

    big_table=big_table.iloc[:, np.r_[0:4,16:31,31, 14:16]]
    big_table = big_table.rename(columns={'isemzyme': 'isenzyme_groundtruth',
                                            'functionCounts': 'functionCounts_groundtruth',
                                            'ec_number': 'ec_groundtruth',
                                            'pred_isEnzyme': 'isenzyme_islice',
                                            'pred_functionCounts': 'functionCounts_islice',
                                            'isemzyme_blast':'isenzyme_blast'
                                            })  
    # 拼合训练测试样本数
    samplecounts = pd.read_csv(cfg.DATADIR + 'ecsamplecounts.tsv', sep = '\t')

    big_table = big_table.merge(samplecounts, left_on='ec_groundtruth', right_on='ec_number', how='left')
    big_table = big_table.iloc[:, np.r_[0:22,23:25]]      

    big_table.to_excel(cfg.FILE_EVL_RESULTS, index=None)
    big_table = big_table.drop_duplicates(subset='id')
    return big_table

def caculateMetrix(groundtruth, predict, baselineName, type='binary'):
    
    if type == 'binary':
        acc = metrics.accuracy_score(groundtruth, predict)
        precision = metrics.precision_score(groundtruth, predict, zero_division=True )
        recall = metrics.recall_score(groundtruth, predict,  zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, zero_division=True)
        tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()
        npv = tn/(fn+tn+1.4E-45)
        print(baselineName, '\t\t%f' %acc,'\t%f'% precision,'\t\t%f'%npv,'\t%f'% recall,'\t%f'% f1, '\t', 'tp:',tp,'fp:',fp,'fn:',fn,'tn:',tn)
    
    if type =='include_unfind':
        evadf = pd.DataFrame()
        evadf['g'] = groundtruth
        evadf['p'] = predict

        evadf_hot = evadf[~evadf.p.isnull()]
        evadf_cold = evadf[evadf.p.isnull()]

        tp = len(evadf_hot[(evadf_hot.g.astype('int')==1) & (evadf_hot.p.astype('int')==1)])
        fp = len(evadf_hot[(evadf_hot.g.astype('int')==0) & (evadf_hot.p.astype('int')==1)])        
        tn = len(evadf_hot[(evadf_hot.g.astype('int')==0) & (evadf_hot.p.astype('int')==0)])
        fn = len(evadf_hot[(evadf_hot.g.astype('int')==1) & (evadf_hot.p.astype('int')==0)])
        up = len(evadf_cold[evadf_cold.g==1])
        un = len(evadf_cold[evadf_cold.g==0])
        acc = (tp+tn)/(tp+fp+tn+fn+up+un)
        precision = tp/(tp+fp)
        npv = tn/(tn+fn)
        recall = tp/(tp+fn+up)
        f1=(2*precision*recall)/(precision+recall)
        print(  baselineName, 
                '\t\t%f' %acc,
                '\t%f'% precision,
                '\t\t%f'%npv,
                '\t%f'% recall,
                '\t%f'% f1, '\t', 
                'tp:',tp,'fp:',fp,'fn:',fn,'tn:',tn, 'up:',up, 'un:',un)
    
    if type == 'multi':
        acc = metrics.accuracy_score(groundtruth, predict)
        precision = metrics.precision_score(groundtruth, predict, average='macro', zero_division=True )
        recall = metrics.recall_score(groundtruth, predict, average='macro', zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, average='macro', zero_division=True)
        print('%12s'%baselineName, ' \t\t%f '%acc,'\t%f'% precision, '\t\t%f'% recall,'\t%f'% f1)

def evalueate_performance(evalutation_table):
    print('\n\n1. isEnzyme prediction evalueation metrics')
    print('*'*140+'\n')
    print('baslineName', '\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', '\t confusion Matrix')
    ev_isenzyme={
                'ours':'isenzyme_islice', 
                'blast':'isenzyme_blast', 
                'ecpred':'isemzyme_ecpred', 
                'deepec':'isemzyme_deepec', 
                'catfam':'isenzyme_catfam',
                'priam':'isenzyme_priam'
                }

    for k,v in ev_isenzyme.items():
        caculateMetrix( groundtruth=evalutation_table.isenzyme_groundtruth.astype('int'), 
                        predict=evalutation_table[v], 
                        baselineName=k, 
                        type='include_unfind')


    print('\n\n2. EC prediction evalueation metrics')
    print('*'*140+'\n')
    print('%12s'%'baslineName', '\t\t', 'accuracy','\t', 'precision-macro \t', 'recall-macro','\t', 'f1-macro')
    ev_ec = {
                'ours':'ec_islice', 
                'blast':'ec_blast', 
                'ecpred':'ec_ecpred', 
                'deepec':'ec_deepec',
                'catfam':'ec_catfam',
                'priam':'ec_priam',
                }
    for k, v in ev_ec.items():
        caculateMetrix( groundtruth=evalutation_table.ec_groundtruth, 
                predict=evalutation_table[v].fillna('NaN'), 
                baselineName=k, 
                type='multi')

    print('\n\n3. Function counts evalueation metrics')
    print('*'*140+'\n')
    print('%12s'%'baslineName', '\t\t', 'accuracy','\t', 'precision-macro \t', 'recall-macro','\t', 'f1-macro')

    ev_functionCounts = {
                            'ours':'functionCounts_islice', 
                            'blast':'functionCounts_blast',  
                            'ecpred':'functionCounts_ecpred',
                            'deepec':'functionCounts_deepec' , 
                            'catfam':'functionCounts_catfam',
                            'priam': 'functionCounts_priam'
                        }

    for k, v in ev_functionCounts.items():
        caculateMetrix( groundtruth=evalutation_table.functionCounts_groundtruth, 
                predict=evalutation_table[v].fillna('-1').astype('int'), 
                baselineName=k, 
                type='multi')

    num_enzyme = len(evalutation_table[evalutation_table.isenzyme_groundtruth])
    num_no_enzyme = len(evalutation_table[~evalutation_table.isenzyme_groundtruth])
    num_multi_function = len(evalutation_table[evalutation_table.functionCounts_groundtruth>1])
    num_single_function = len(evalutation_table[evalutation_table.functionCounts_groundtruth==1])


    ev_ec = {'ours':'ec_islice', 'blast':'ec_blast', 'ecpred':'ec_ecpred', 'deepec':'ec_deepec', 'catfam':'ec_catfam'}
    
    print('\n\n4. EC Prediction Report')
    str_out_head = '\t item'
    str_out_multi = '多功能酶的accuracy'
    str_out_single = '单功能酶的accuracy'
    str_out_all ='整体accuracy\t'
    for k, v in ev_ec.items():
        str_out_head += ('\t\t\t'+str(k))
        num_multi = len(evalutation_table[(evalutation_table.functionCounts_groundtruth>1) &
                                            (evalutation_table.ec_groundtruth == evalutation_table[v])])
        num_single = len(evalutation_table[(evalutation_table.functionCounts_groundtruth==1) &
                                            (evalutation_table.ec_groundtruth == evalutation_table[v])])
        str_out_multi  +=  ('\t\t' + str(num_multi)+'/'+ str(num_multi_function) +'='+ str(round(num_multi/num_multi_function,4)))
        str_out_single += ('\t\t' + str(num_single)+'/'+ str(num_single_function) +'='+ str(round(num_single/num_single_function,4)))
        str_out_all += ('\t\t' + str(num_single + num_multi)+'/'+ str(num_enzyme) +'='+ str(round(num_single/num_enzyme,4)))
    print(str_out_head )
    print(str_out_multi)
    print(str_out_single)
    print(str_out_all)

def filter_newadded_ec(restable):
    uctools =  ucTools.ucTools('172.16.25.20')
    cnx_mimic = uctools.db_conn()
    sql ='select * from train_test_common_ec'
    cmec = pd.read_sql_query(sql,cnx_mimic)
    restable = restable[
                (restable.ec_groundtruth.isin(list(cmec.ec_number))) | 
                (~restable.isenzyme_groundtruth) | 
                (restable.functionCounts_groundtruth>1)
        ]
    restable.reset_index(drop=True, inplace=True)

    return restable

if __name__ =='__main__':

    # 1. 读入数据
    train = pd.read_feather(cfg.DATADIR+'train.feather').iloc[:,:6]
    test = pd.read_feather(cfg.DATADIR+'test.feather').iloc[:,:6]
    # test = test[(test.ec_specific_level>=cfg.TRAIN_USE_SPCIFIC_EC_LEVEL) |(~test.isemzyme)]
    test.reset_index(drop=True, inplace=True)
    # EC-标签字典
    dict_ec_label = np.load(cfg.DATADIR + 'ec_label_dict.npy', allow_pickle=True).item()
    file_blast_res = cfg.RESULTSDIR + r'test_blast_res.tsv'
    flat_table = load_res_data(
        file_slice=cfg.FILE_INTE_RESULTS,
        file_blast=cfg.FILE_BLAST_RESULTS,
        file_deepec=cfg.FILE_DEEPEC_RESULTS,
        file_ecpred=cfg.FILE_ECPRED_RESULTS,
        file_catfam = cfg.FILE_CATFAM_RESULTS,
        file_priam = cfg.FILE_PRIAM_RESULTS,
        train=train,
        test=test
        )

    evalutation_table = integrate_reslults(flat_table)
    evalutation_table = filter_newadded_ec(evalutation_table)
    
    evalueate_performance(evalutation_table)
    evalutation_table['sright'] = evalutation_table.apply(lambda x: True if x.ec_groundtruth == x.ec_islice else False, axis=1)
    evalutation_table['bright'] = evalutation_table.apply(lambda x: True if x.ec_groundtruth == x.ec_blast else False, axis=1)
    evalutation_table.to_excel(cfg.RESULTSDIR+'evaluationFF.xlsx', index=None)  
    print('\n Evaluation Finished \n\n')