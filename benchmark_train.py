import pandas as pd
import numpy as np
from tkinter import _flatten
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from tqdm import tqdm
import benchmark_common as bcommon
import config as cfg
import os

#region 获取酶训练的数据集
def get_train_X_Y(traindata, feature_bankfile, task=1, task3_test=None):
    """[获取酶训练的数据集]

    Args:
        traindata ([DataFrame]): [description]

    Returns:
        [DataFrame]: [trianX, trainY]
    """
    traindata = traindata.merge(feature_bankfile, on='id', how='left')
    train_X = traindata.iloc[:,3:]
    if task == 1:
        train_Y = traindata['isenzyme'].astype('int')
    if task == 2:
        train_Y = traindata['functionCounts'].astype('int')
    if task == 3 :
        dict_ec_label  = make_ec_label_dict(ec_str_list=list(set(traindata.ec_number.append(data_task3_test.ec_number).values)))
        task3_data = []
        for index, row in tqdm(traindata.iterrows()):
            ec_array = row.ec_number.split(',')
            if len(ec_array) <2:
                task3_data = task3_data + [list(row.values)]
            else:
                for item in ec_array:
                    temp_row = row.copy()
                    temp_row['ec_number'] = item
                    task3_data = task3_data + [list(temp_row.values)]
        
        traindata = pd.DataFrame(task3_data, columns=traindata.columns)
        train_X = traindata.iloc[:,3:]
        train_Y = traindata['ec_number'].apply(lambda x: dict_ec_label.get(x))
    return train_X, train_Y
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
        early_stop = EarlyStopping( rounds=2, 
                            save_best=True,
                            maximize=False,
                            metric_name='auc'
                        )
        model = XGBClassifier(
            objective='binary:logistic', 
            random_state=42, 
            use_label_encoder=False, 
            n_jobs=-2, 
            eval_metric='auc',
            max_depth=6,
            n_estimators= cfg.TRAIN_ISENZYME_LEARNING_STEPS
            )
        
        print(model)
        # model.fit(X, Y.ravel())
        model.fit(x_train, y_train, eval_metric="auc", eval_set=eval_set, verbose=10, callbacks=[early_stop])
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
        early_stop = EarlyStopping( rounds=2, 
                                    save_best=True,
                                    maximize=False,
                                    metric_name='mlogloss'
                                )
        model = XGBClassifier(
                                min_child_weight=6, 
                                max_depth=6, 
                                objective='multi:softmax', 
                                num_class=10, 
                                eval_metric='mlogloss',
                                use_label_encoder=False,
                                n_jobs=-1,
                                n_estimators=cfg.TRAIN_HOWMANY_ENZYME_LEARNING_STEPS
                            )
        print("-" * 100)
        print("几功能酶xgboost模型：", model)
        model.fit(x_train, y_train, eval_metric="mlogloss", eval_set=eval_set,callbacks=[early_stop], verbose=10)
        # # 打印重要性指数
        # bcommon.importance_features_top(model, x_train, topN=50)
        # 保存模型
        joblib.dump(model, model_file)
        return model
#endregion


def make_ec_label_dict(ec_str_list, file_save=cfg.FILE_EC_LABEL_DICT):
    if os.path.exists(file_save):
        print(f'ec label dict already exist, using existing file:{cfg.FILE_EC_LABEL_DICT}')
        dict_ec_label = np.load(cfg.FILE_EC_LABEL_DICT, allow_pickle=True).item()
        return dict_ec_label

    ecs = _flatten([item.split(',') for item in ec_str_list])
    ecs = list(set(ecs))
    ec_label_dict = {k: v for k, v in zip(ecs, range(len(ecs)))}
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

    EMBEDDING_METHOD = 'esm32'

    # 1. 读入数据
    print('step 1 loading task data')
    data_task1_train = pd.read_feather(cfg.FILE_TASK1_TRAIN)
    data_task2_train = pd.read_feather(cfg.FILE_TASK2_TRAIN)
    data_task3_train = pd.read_feather(cfg.FILE_TASK3_TRAIN)
    data_task3_test = pd.read_feather(cfg.FILE_TASK3_TEST)

    # 2. 读取特征
    print(f'step 2: Loading features, embdding method={EMBEDDING_METHOD}')
    feature_df = bcommon.load_data_embedding(embedding_type=EMBEDDING_METHOD)

    #3. task1 train
    print('step 3: train isEnzyme model')
    task1_X, task1_Y = get_train_X_Y(traindata=data_task1_train, feature_bankfile=feature_df, task=1)
    train_isenzyme(X=task1_X, Y=task1_Y, model_file= cfg.ISENZYME_MODEL, force_model_update=cfg.UPDATE_MODEL)


    #4. task2 train
    print('step 4: train how many enzymes model')
    task2_X, task2_Y = get_train_X_Y(traindata=data_task2_train, feature_bankfile=feature_df, task=2)
    train_howmany_enzyme(data_x=task2_X, data_y=(task2_Y-1), model_file=cfg.HOWMANY_MODEL, force_model_update=cfg.UPDATE_MODEL)


    #4. 加载EC号训练数据
    print('loading ec to label dict')
    task3_X, task3_Y = get_train_X_Y(traindata=data_task3_train, feature_bankfile=feature_df, task=3, task3_test=data_task3_test)
    print('step 4 prepare slice files')
    #5. 构建Slice文件
    dict_ec_label = np.load(cfg.FILE_EC_LABEL_DICT, allow_pickle=True).item()
    bcommon.prepare_slice_file(x_data=task3_X, y_data=task3_Y, x_file=cfg.FILE_SLICE_TRAINX, y_file=cfg.FILE_SLICE_TRAINY, ec_label_dict=dict_ec_label)
    
    print('step 6 trainning slice model')
    #6. 训练Slice模型
    train_ec_slice(trainX=cfg.FILE_SLICE_TRAINX, trainY=cfg.FILE_SLICE_TRAINY, modelPath=cfg.MODELDIR)
    

    # # 2. 写入fasta文件

    # table2fasta(table=train,file_out=cfg.TRAIN_FASTA)
    # table2fasta(table=test, file_out=cfg.TEST_FASTA)
    # #7. 创建blast比对数据库
    # bcommon.make_diamond_db(dbtable=train.iloc[:, np.r_[0,5]],to_db_file=cfg.FILE_BLAST_TRAIN_DB) # 创建是否是酶blast数据库

    print('train finished')