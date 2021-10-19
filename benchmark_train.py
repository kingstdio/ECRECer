from json import load
from pickle import FALSE
from tools.funclib import table2fasta
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import benchmark_common as bcommon
import config as cfg
import os

#region 获取酶训练的数据集
def get_enzyme_train_set(traindata):
    """[获取酶训练的数据集]

    Args:
        traindata ([DataFrame]): [description]

    Returns:
        [DataFrame]: [trianX, trainY]
    """
    train_X = traindata.iloc[:,7:]
    train_Y = traindata['isemzyme'].astype('int')
    return train_X, train_Y
#endregion

#region 获取几功能酶训练的数据集
def get_howmany_train_set(train_data):
    """[获取几功能酶训练的数据集]
    Args:
        train_data ([DataFrame]): [完整训练数据]
    Returns:
        [DataFrame]: [[train_x, trian_y]]
    """
    train_data = train_data[train_data.isemzyme] #仅选用酶数据
    train_X = train_data.iloc[:,7:]
    train_Y =train_data['functionCounts'].astype('int')
    return train_X, pd.DataFrame(train_Y)

#endregion

#region 获取EC号训练的数据集
def get_ec_train_set(train_data, ec_label_dict):
    """[获取EC号训练的数据集]
    Args:
        ec_label_dict: ([dict]): [ec to label dict]
        train_data ([DataFrame]): [description]
    Returns:
        [DataFrame]: [[train_x, trian_y]]
    """
    if cfg.TRAIN_USE_ONLY_ENZYME:
        train_data = train_data[train_data.isemzyme] #仅选用酶数据
    # if cfg.TRAIN_USE_ONLY_SINGLE_FUNCTION:
    train_data = train_data[train_data.functionCounts ==1] #仅选用单功能酶数据
    
    train_data = train_data[(train_data.ec_specific_level >= cfg.TRAIN_USE_SPCIFIC_EC_LEVEL) |(train_data.ec_specific_level ==0)]
    train_data.reset_index(drop=True, inplace=True)
    train_data.insert(loc=1, column='ec_label', value=train_data.ec_number.apply(lambda x: ec_label_dict.get(x)))
    # train_data['ec_label'] = train_data.ec_number.apply(lambda x: ec_label_dict.get(x))

    train_X = train_data.iloc[:, 8:]
    train_Y =train_data['ec_label']
    return train_X, pd.DataFrame(train_Y)

#endregion

#region 训练是否是酶模型
def train_isenzyme(X,Y, model_file, vali_ratio=0.3, force_model_update=False):
    """[训练是否是酶模型]

    Args:
        vali_ratio:
        X ([DataFrame]): [特征数据]
        Y ([DataFrame]): [标签数据]
        model_file ([string]): [模型的存放路径]
        force_model_update (bool, optional): [是否强制更新模型]. Defaults to False.

    Returns:
        [object]: [训练好的模型]
    """
    if os.path.exists(model_file) and (force_model_update==False):
        return
    else:
        x_train, x_vali, y_train, y_vali = train_test_split(X,np.array(Y).ravel(),test_size=vali_ratio,random_state=1)
        eval_set = [(x_train, y_train), (x_vali, y_vali)]

        model = XGBClassifier(
            objective='binary:logistic', 
            random_state=42, 
            use_label_encoder=False, 
            n_jobs=-2, 
            eval_metric='mlogloss',
            max_depth=6,
            n_estimators= cfg.TRAIN_HOWMANY_ENZYME_LEARNING_STEPS
            )
        
        print(model)
        # model.fit(X, Y.ravel())
        model.fit(x_train, y_train, eval_metric="logloss", eval_set=eval_set, verbose=True)
        joblib.dump(model, model_file)
        print('XGBoost模型训练完成')
        return model
#endregion

#region 构建几功能酶模型
def train_howmany_enzyme(data_x, data_y, model_file, vali_ratio=0.3, force_model_update=False):
    """[构建几功能酶模型]

    Args:
        force_model_update:
        vali_ratio:
        model_file:
        data_x ([DataFrame]): [X训练数据]
        data_y ([DataFrame]): [Y训练数据]

    Returns:
        [object]: [训练好的模型]
    """
    if os.path.exists(model_file) and (force_model_update==False):
        return
    else:
        x_train, x_vali, y_train, y_vali = train_test_split(data_x,np.array(data_y).ravel(),test_size=vali_ratio,random_state=1)
        eval_set = [(x_train, y_train), (x_vali, y_vali)]
        
        model = XGBClassifier(
                                min_child_weight=6, 
                                max_depth=6, 
                                objective='multi:softmax', 
                                num_class=10, 
                                use_label_encoder=False,
                                n_estimators=cfg.TRAIN_HOWMANY_ENZYME_LEARNING_STEPS
                            )
        print("-" * 100)
        print("几功能酶xgboost模型：", model)
        model.fit(x_train, y_train, eval_metric="mlogloss", eval_set=eval_set, verbose=False)
        # # 打印重要性指数
        bcommon.importance_features_top(model, x_train, topN=50)
        # 保存模型
        joblib.dump(model, model_file)
        return model
#endregion

def make_ec_label(train_label, test_label, file_save, force_model_update=False):
    if os.path.exists(file_save) and (force_model_update==False):
        print('ec label dict already exist')
        return
    ecset = sorted( set(list(train_label) + list(test_label)))
    ec_label_dict = {k: v for k, v in zip(ecset, range(len(ecset)))}
    np.save(file_save, ec_label_dict)
    return  ec_label_dict
    print('字典保存成功')


#region 训练slice模型
def train_ec_slice(trainX, trainY, modelPath, force_model_update=False):
    """[训练slice模型]
    Args:
        trainX ([DataFarame]): [X特征]
        trainY ([DataFrame]): [ Y标签]]
        modelPath ([string]): [存放模型的目录]
        force_model_update (bool, optional): [是否强制更新模型]. Defaults to False.
    """
    if os.path.exists(modelPath+'param') and (force_model_update==False):
        print('model exist')
        return

    cmd = ''' ./slice_train {0} {1} {2} -m 100 -c 300 -s 300 -k 700 -o 32 -t 32 -C 1 -f 0.000001 -siter 20 -stype 0 -q 0 '''.format(trainX, trainY, modelPath)
    print(cmd)
    os.system(cmd)
    print('train finished')
#endregion





if __name__ =="__main__":

    # 1. 读入数据
    print('step 1 loading data')
    train = pd.read_feather(cfg.TRAIN_FEATURE)
    test = pd.read_feather(cfg.TEST_FEATURE)

    train,test= bcommon.load_data_embedding(train=train, test=test, embedding_type=cfg.EMBEDDING_METHOD.get('unirep'))

    # # 2. 写入fasta文件

    table2fasta(table=train,file_out=cfg.TRAIN_FASTA)
    table2fasta(table=test, file_out=cfg.TEST_FASTA)

    #2. 「酶｜非酶」模型训练
    print('step 2 train isEnzyme model')
    enzyme_X, enzyme_Y = get_enzyme_train_set(train)
    train_isenzyme(X=enzyme_X, Y=enzyme_Y, model_file= cfg.ISENZYME_MODEL, force_model_update=cfg.UPDATE_MODEL)


    #3. 「几功能酶训练模型」训练
    print('step 3 train how many enzymes model')
    howmany_X, howmany_Y = get_howmany_train_set(train)
    train_howmany_enzyme(data_x=howmany_X, data_y=(howmany_Y-1), model_file=cfg.HOWMANY_MODEL, force_model_update=cfg.UPDATE_MODEL)

    #4. 加载EC号训练数据
    print('loading ec to label dict')
    if os.path.exists(cfg.FILE_EC_LABEL_DICT):
        dict_ec_label = np.load(cfg.FILE_EC_LABEL_DICT, allow_pickle=True).item()
    else:
        dict_ec_label = make_ec_label(train_label=train['ec_number'], test_label=test['ec_number'], file_save= cfg.FILE_EC_LABEL_DICT, force_model_update=cfg.UPDATE_MODEL)

    ec_X, ec_Y = get_ec_train_set(train_data=train, ec_label_dict=dict_ec_label)

    print('step 4 prepare slice files')
    #5. 构建Slice文件
    bcommon.prepare_slice_file(x_data=ec_X, y_data=ec_Y, x_file=cfg.FILE_SLICE_TRAINX, y_file=cfg.FILE_SLICE_TRAINY, ec_label_dict=dict_ec_label)
    
    print('step 6 trainning slice model')
    #6. 训练Slice模型
    train_ec_slice(trainX=cfg.FILE_SLICE_TRAINX, trainY=cfg.FILE_SLICE_TRAINY, modelPath=cfg.MODELDIR)

    #7. 创建blast比对数据库
    bcommon.make_diamond_db(dbtable=train.iloc[:, np.r_[0,5]],to_db_file=cfg.FILE_BLAST_TRAIN_DB) # 创建是否是酶blast数据库

    print('train finished')