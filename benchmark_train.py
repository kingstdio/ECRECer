import pandas as pd
import numpy as np
import joblib,os,sys
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import benchmark_common as bcommon
import config as cfg
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense
from keras.layers import GRU, Bidirectional
from tools import Attention
from keras.callbacks import TensorBoard,ModelCheckpoint
import tools.funclib as fuclib
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def make_onehot_label(label_list, save=False, file_encoder='./encode.h5', type='singel'):
    if type=='singel':
        encoder = preprocessing.OneHotEncoder(sparse=False)
        results = encoder.fit_transform([[item] for item in label_list])
    elif type=='multi':
        label_list = data_task3_train.ec_number.to_list()
        encoder = preprocessing.MultiLabelBinarizer()
        results=encoder.fit_transform([ item.split(',') for item in label_list])
    else:
        print('lable encoding type error, please check')
        sys.exit()

    if save ==True:
        joblib.dump(encoder, file_encoder)

    return results


#region 获取酶训练的数据集
def get_train_X_Y(traindata, feature_bankfile, task=1):
    """[获取酶训练的数据集]

    Args:
        traindata ([DataFrame]): [description]

    Returns:
        [DataFrame]: [trianX, trainY]
    """
    traindata = traindata.merge(feature_bankfile, on='id', how='left')
    train_X = np.array(traindata.iloc[:,3:])
    if task == 1:
        train_Y = make_onehot_label(label_list=traindata['isenzyme'].to_list(), save=True, file_encoder=cfg.DICT_LABEL_T1,type='singel')
    if task == 2:
        train_Y = make_onehot_label(label_list=traindata['functionCounts'].to_list(), save=True, file_encoder=cfg.DICT_LABEL_T2, type='singel')
    if task == 3 :
        train_Y = make_onehot_label(label_list=traindata['ec_number'].to_list(), save=True, file_encoder=cfg.DICT_LABEL_T3, type='multi')
    return train_X, train_Y
#endregion


def mgru_attion_model(input_dimensions, gru_h_size=512,  dropout=0.2,  lossfunction='binary_crossentropy', 
                      evaluation_metrics='accuracy', activation_method = 'sigmoid', output_dimensions=2
                    ):
    inputs = Input(shape=(1, input_dimensions), name="input")
    gru = Bidirectional(GRU(gru_h_size, dropout=dropout, return_sequences=True), name="bi-gru")(inputs)
    attention = Attention.Attention(32)(gru)
    output = Dense(output_dimensions, activation=activation_method, name="dense")(attention)
    model = Model(inputs, output)
    model.compile(loss=lossfunction, optimizer=Adam(),metrics=[evaluation_metrics])

    return model


#region 训练是否是酶模型
def train_isenzyme(X,Y, model_file, vali_ratio=0.3, force_model_update=False, epochs=1):
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
        x_train, x_vali, y_train, y_vali = train_test_split(np.array(X), np.array(Y), test_size=vali_ratio, shuffle=True)
        x_train = x_train.reshape(x_train.shape[0],1,-1)
        x_vali  = x_vali.reshape(x_vali.shape[0],1,-1)
        tbCallBack = TensorBoard(log_dir=f'{cfg.TEMPDIR}model/task1', histogram_freq=1,write_grads=True)
        checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy', mode='auto', save_best_only='True')
        instant_model = mgru_attion_model(input_dimensions=X.shape[1], gru_h_size=512, dropout=0.3, lossfunction='binary_crossentropy', 
                        evaluation_metrics='accuracy', activation_method = 'sigmoid', output_dimensions=Y.shape[1] )
        
        instant_model.fit(x_train, y_train, validation_data=(x_vali, y_vali), batch_size=512, epochs= epochs, callbacks=[tbCallBack, checkpoint])
        # 保存
        print(f'train_isenzyme finished, best model saved to: {model_file}')
        
#endregion

#region 构建几功能酶模型
def train_howmany_enzyme(X, Y, model_file, force_model_update=False, epochs=1):
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
        x_train, x_vali, y_train, y_vali = train_test_split(np.array(X), np.array(Y), test_size=0.3, shuffle=True)
        x_train = x_train.reshape(x_train.shape[0],1,-1)
        x_vali  = x_vali.reshape(x_vali.shape[0],1,-1)
        tbCallBack = TensorBoard(log_dir=f'{cfg.TEMPDIR}model/task1', histogram_freq=1,write_grads=True)
        checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy', mode='auto', save_best_only='True')
        instant_model = mgru_attion_model(input_dimensions=X.shape[1], gru_h_size=512, dropout=0.2, lossfunction='categorical_crossentropy', 
                        evaluation_metrics='accuracy', activation_method = 'softmax', output_dimensions=Y.shape[1])
        
        instant_model.fit(x_train, y_train, validation_data=(x_vali, y_vali), batch_size=512, epochs= epochs, callbacks=[tbCallBack, checkpoint])
        
        print(f'train how many enzyme finished, best model saved to: {model_file}')
#endregion

def train_ec(X, Y, model_file,  force_model_update=False, epochs=1):
    if os.path.exists(model_file) and (force_model_update==False):
        return
    else:
        x_train, x_vali, y_train, y_vali = train_test_split(np.array(X), np.array(Y), test_size=0.3, shuffle=True)
        x_train = x_train.reshape(x_train.shape[0],1,-1)
        x_vali  = x_vali.reshape(x_vali.shape[0],1,-1)
        instant_model = mgru_attion_model(input_dimensions=X.shape[1], gru_h_size=512, dropout=0.2, lossfunction='categorical_crossentropy', 
                        evaluation_metrics='accuracy', activation_method = 'softmax', output_dimensions=Y.shape[1] )
        tbCallBack = TensorBoard(log_dir=f'{cfg.TEMPDIR}model/task1', histogram_freq=1,write_grads=True)
        checkpoint = ModelCheckpoint(filepath=model_file, monitor='val_accuracy', mode='auto', save_best_only='True')
        instant_model.fit(x_train, y_train, validation_data=(x_vali, y_vali), batch_size=3948, epochs= epochs, callbacks=[tbCallBack, checkpoint])
        # 保存
        
        print(f'train EC model finished, best model saved to: {model_file}')



if __name__ =="__main__":

    EMBEDDING_METHOD = 'esm32'

    # 1. read tranning data
    print('step 1 loading task data')

    # please specific the tranning data for each tasks
    data_task1_train = pd.read_feather(cfg.FILE_TASK1_TRAIN)
    data_task2_train = pd.read_feather(cfg.FILE_TASK2_TRAIN)
    data_task3_train = pd.read_feather(cfg.FILE_TASK3_TRAIN)


    # 2. read tranning feature
    print(f'step 2: Loading features, embdding method={EMBEDDING_METHOD}')
    feature_df = bcommon.load_data_embedding(embedding_type=EMBEDDING_METHOD)

    #3. train task-1 model
    print('step 3: train isEnzyme model')
    task1_X, task1_Y = get_train_X_Y(traindata=data_task1_train, feature_bankfile=feature_df, task=1)
    train_isenzyme(X=task1_X, Y=task1_Y, model_file= cfg.ISENZYME_MODEL, force_model_update=cfg.UPDATE_MODEL, epochs=1000)

    #4. task2 train
    print('step 4: train how many enzymes model')
    task2_X, task2_Y = get_train_X_Y(traindata=data_task2_train, feature_bankfile=feature_df, task=1000)
    train_howmany_enzyme(X=task2_X, Y=task2_Y, model_file=cfg.HOWMANY_MODEL, force_model_update=cfg.UPDATE_MODEL, epochs=2)

    #5. task3 train
    print('step 5 train EC model')
    task3_X, task3_Y = get_train_X_Y(traindata=data_task3_train, feature_bankfile=feature_df, task=3)
    train_ec(X=task3_X, Y=task3_Y, model_file=cfg.EC_MODEL, force_model_update=cfg.UPDATE_MODEL, epochs=1000)

    # 6. 写入fasta文件
    fuclib.table2fasta(table=data_task1_train,file_out=cfg.TRAIN_FASTA)
    #7. 创建blast比对数据库
    bcommon.make_diamond_db(dbtable=data_task1_train.iloc[:, np.r_[0,5]],to_db_file=cfg.FILE_BLAST_TRAIN_DB) # 创建酶blast数据库

    print('train finished')