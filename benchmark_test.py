import re
import pandas as pd
import numpy as np
import joblib
import os
import benchmark_common as bcommon
import config as cfg
from Bio import SeqIO

# region 获取「酶｜非酶」预测结果
def get_isEnzymeRes(querydata, model_file):
    """[获取「酶｜非酶」预测结果]
    Args:
        querydata ([DataFrame]): [需要预测的数据]
        model_file ([string]): [模型文件]
    Returns:
        [DataFrame]: [预测结果、预测概率]
    """
    model = joblib.load(model_file)
    predict = model.predict(querydata)
    predictprob = model.predict_proba(querydata)
    return predict, predictprob[:, 1]
# endregion

# region 获取「几功能酶」预测结果
def get_howmany_Enzyme(querydata, model_file):
    """获取「几功能酶」预测结果
    Args:
        querydata ([DataFrame]): [需要预测的数据]
        model_file ([string]): [模型文件]
    Returns:
        [DataFrame]: [预测结果、预测概率]
    """
    model = joblib.load(model_file)
    predict = model.predict(querydata)
    predictprob = model.predict_proba(querydata)
    return predict+1, predictprob #标签加1，单功能酶标签为0，统一加1
# endregion

# region 获取slice预测结果
def get_slice_res(slice_query_file, model_path, dict_ec_label,test_set, res_file):
    """[获取slice预测结果]

    Args:
        slice_query_file ([string]): [需要预测的数据sliceFile]
        model_path ([string]): [Slice模型路径]
        res_file ([string]]): [预测结果文件]
    Returns:
        [DataFrame]: [预测结果]
    """

    cmd = '''./slice_predict {0} {1} {2} -o 32 -b 0 -t 32 -q 0'''.format(slice_query_file, model_path, res_file)
    print(cmd)
    os.system(cmd)
    result_slice = pd.read_csv(res_file, header=None, skiprows=1, sep=' ')

        # 5.3 对预测结果排序
    slice_pred_rank, slice_pred_prob = sort_results(result_slice)

    # 5.4 将结果翻译成EC号
    slice_pred_ec = translate_slice_pred(slice_pred=slice_pred_rank, ec2label_dict = dict_ec_label, test_set=test_set)

    return slice_pred_ec

# endregion

#region 将slice的实验结果排序，并按照推荐顺序以两个矩阵的形式返回
def sort_results(result_slice):
    """
    将slice的实验结果排序，并按照推荐顺序以两个矩阵的形式返回
    @pred_top：预测结果排序
    @pred_pb_top：预测结果评分排序
    """
    pred_top = []
    pred_pb_top = []
    aac = []
    for index, row in result_slice.iterrows():
        row_trans = [*row.apply(lambda x: x.split(':')).values]
        row_trans = pd.DataFrame(row_trans).sort_values(by=[1], ascending=False)
        pred_top += [list(np.array(row_trans[0]).astype('int'))]
        pred_pb_top += [list(np.array(row_trans[1]).astype('float'))]
    pred_top = pd.DataFrame(pred_top)
    pred_pb_top = pd.DataFrame(pred_pb_top)
    return pred_top, pred_pb_top
#endregion

#region 划分测试集XY
def get_test_set(data):
    """[划分测试集XY]

    Args:
        data ([DataFrame]): [测试数据]

    Returns:
        [DataFrame]: [划分好的XY]
    """
    testX = data.iloc[:,7:]
    testY = data.iloc[:,:6]
    return testX, testY
#endregion

#region 将slice预测的标签转换为EC号
def translate_slice_pred(slice_pred, ec2label_dict, test_set):
    """[将slice预测的标签转换为EC号]

    Args:
        slice_pred ([DataFrame]): [slice预测后的排序数据]
        ec2label_dict ([dict]]): [ec转label的字典]
        test_set ([DataFrame]): [测试集用于取ID]

    Returns:
        [type]: [description]
    """
    label_ec_dict = {value:key for key, value in ec2label_dict.items()}
    res_df = pd.DataFrame()
    res_df['id'] = test_set.id
    colNames = slice_pred.columns.values
    for colName in colNames:
        res_df['top'+str(colName)] = slice_pred[colName].apply(lambda x: label_ec_dict.get(x))
    return res_df
#endregion



#region 将测试结果进行集成输出
def run_integrage(slice_pred, dict_ec_transfer):
    """[将测试结果进行集成输出]

    Args:
        slice_pred ([DataFrame]): [slice 预测结果]
        dict_ec_transfer ([dict]): [EC转移dict]

    Returns:
        [DataFrame]: [集成后的最终结果]
    """
    # 取top10,因为最多有10功能酶
    slice_pred = slice_pred.iloc[:,np.r_[0:11, 21:28]]

    #酶集成标签
    with pd.option_context('mode.chained_assignment', None):
        slice_pred['is_enzyme_i'] = slice_pred.apply(lambda x: int(x.isemzyme_blast) if str(x.isemzyme_blast)!='nan' else x.isEnzyme_pred_xg, axis=1)

    # 清空非酶的EC预测标签
    for i in range(9):
        with pd.option_context('mode.chained_assignment', None):
            slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if x.is_enzyme_i==0 else x['top'+str(i)], axis=1)
            slice_pred['top0'] = slice_pred.apply(lambda x: x.ec_number_blast if str(x.ec_number_blast)!='nan' else x.top0, axis=1)

    # 清空有比对结果的预测标签
    for i in range(1,10):
        with pd.option_context('mode.chained_assignment', None):
            slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if str(x.ec_number_blast)!='nan' else x['top'+str(i)], axis=1) #有比对结果的
            slice_pred['top'+str(i)] = slice_pred.apply(lambda x: '' if int(x.functionCounts_pred_xg) < int(i+1) else x['top'+str(i)], axis=1) #无比对结果的
    with pd.option_context('mode.chained_assignment', None):
        slice_pred['top0']=slice_pred['top0'].apply(lambda x: '' if x=='-' else x)
    
    # 将EC号拆开
    for index, row in slice_pred.iterrows():
        ecitems=row['top0'].split(',')
        if len(ecitems)>1:
            for i in range(len(ecitems)):
                slice_pred.loc[index,'top'+str(i)] =  ecitems[i].strip()

    slice_pred.reset_index(drop=True, inplace=True)

    # 添加几功能酶预测结果
    with pd.option_context('mode.chained_assignment', None):
        slice_pred['pred_functionCounts'] = slice_pred.apply(lambda x: int(x['functionCounts_blast']) if str(x['functionCounts_blast'])!='nan' else x.functionCounts_pred_xg ,axis=1)
    # 取最终的结果并改名
    colnames=[  'id', 
                'pred_ec1', 
                'pred_ec2', 
                'pred_ec3', 
                'pred_ec4', 
                'pred_ec5' , 
                'pred_ec6', 
                'pred_ec7', 
                'pred_ec8', 
                'pred_ec9', 
                'pred_ec10',
                'pred_isEnzyme', 
                'pred_functionCounts'
            ]
    slice_pred=slice_pred.iloc[:, np.r_[0:11, 18,19]]
    slice_pred.columns = colnames

    # 计算EC转移情况
    for i in range(1,11):
        slice_pred['pred_ec'+str(i)] = slice_pred['pred_ec'+str(i)].apply(lambda x: dict_ec_transfer.get(x) if x in dict_ec_transfer.keys() else x)
    
    # 清空没有EC号预测的酶功能数
    with pd.option_context('mode.chained_assignment', None):
        slice_pred.pred_functionCounts[slice_pred.pred_ec1.isnull()] = 0

    return slice_pred
#endregion

if __name__ == '__main__':

    EMBEDDING_METHOD = 'esm32'
    TESTSET='test2019'

    # 1. 读入数据
    print('step 1: loading data')
    train = pd.read_feather(cfg.TRAIN_FEATURE)
    test = pd.read_feather(cfg.TEST_FEATURE)
    train,test= bcommon.load_data_embedding(train=train, test=test, embedding_type=EMBEDDING_METHOD)
    train = train.iloc[:,:7]

    dict_ec_label = np.load(cfg.FILE_EC_LABEL_DICT, allow_pickle=True).item() #EC-标签字典
    dict_ec_transfer = np.load(cfg.FILE_TRANSFER_DICT, allow_pickle=True).item() #EC-转移字典

    # 2. 获取序列比对结果

    print('step 2 get blast results')
    blast_res = bcommon.get_blast_prediction(  reference_db=cfg.FILE_BLAST_TRAIN_DB, 
                                                train_frame=train, 
                                                test_frame=test.iloc[:,0:7],
                                                results_file=cfg.FILE_BLAST_RESULTS,
                                                identity_thres=cfg.TRAIN_BLAST_IDENTITY_THRES
                                            )

    # 3.获取酶-非酶预测结果
    print('step 3. get isEnzyme results')
    testX, testY = get_test_set(data=test)
    isEnzyme_pred, isEnzyme_pred_prob = get_isEnzymeRes(querydata=testX, model_file=cfg.ISENZYME_MODEL)


    # 4. 预测几功能酶预测结果
    print('step 4. get howmany functions ')
    howmany_Enzyme_pred, howmany_Enzyme_pred_prob = get_howmany_Enzyme(querydata=testX, model_file=cfg.HOWMANY_MODEL)

    # 5.获取Slice预测结果
    print('step 5. get EC prediction results')
    # 5.1 准备slice所用文件
    bcommon.prepare_slice_file( x_data=testX, 
                                y_data=testY['ec_number'], 
                                x_file=cfg.FILE_SLICE_TESTX,
                                y_file=cfg.FILE_SLICE_TESTY,
                                ec_label_dict=dict_ec_label
                            )
    # 5.2 获得预测结果
    # slice_pred_ec = get_slice_res(slice_query_file=cfg.FILE_SLICE_TESTX, model_path= cfg.MODELDIR, dict_ec_label=dict_ec_label, test_set=test,  res_file=cfg.FILE_SLICE_RESULTS)
    slice_pred_ec = get_slice_res(slice_query_file=cfg.DATADIR+'slice_test_x_esm33.txt', model_path= cfg.MODELDIR+'/slice_esm33', dict_ec_label=dict_ec_label, test_set=test,  res_file=cfg.FILE_SLICE_RESULTS)
    
    slice_pred_ec['isEnzyme_pred_xg'] = isEnzyme_pred
    slice_pred_ec['functionCounts_pred_xg'] = howmany_Enzyme_pred
    slice_pred_ec = slice_pred_ec.merge(blast_res, on='id', how='left')

    # slice_pred_ec.to_csv(cfg.RESULTSDIR + 'singele_slice.tsv', sep='\t', index=None)
    # 5.5 获取blast EC预测结果

    # 6.将结果集成输出(slice_pred=slice_pred_ec, dict_ec_transfer=dict_ec_transfer)
    slice_pred_ec = run_integrage(slice_pred=slice_pred_ec, dict_ec_transfer = dict_ec_transfer)    
    slice_pred_ec.to_csv(cfg.FILE_INTE_RESULTS, sep='\t', index=None)

    print('predict finished')