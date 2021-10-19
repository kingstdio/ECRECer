from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import linear_model, datasets
from sklearn.svm import SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from xgboost import XGBClassifier
from tqdm import tqdm

import joblib

import pandas as pd
import numpy as np
import os


table_head = [  'id', 
                'name',
                'isemzyme',
                'isMultiFunctional', 
                'functionCounts', 
                'ec_number', 
                'ec_specific_level',
                'date_integraged',
                'date_sequence_update',
                'date_annotation_update',
                'seq', 
                'seqlength'
            ]

def table2fasta(table, file_out):
    file = open(file_out, 'w')
    for index, row in table.iterrows():
        file.write('>{0}\n'.format(row['id']))
        file.write('{0}\n'.format(row['seq']))
    file.close()
    print('Write finished')
    
#氨基酸字典(X未知项用于数据对齐)
prot_dict = dict(
                    A=1,  R=2,  N=3,  D=4,  C=5,  E=6,  Q=7,  G=8,  H=9,  O=10, I=11, L=12, 
                    K=13, M=14, F=15, P=16, U=17, S=18, T=19, W=20, Y=21, V=22, B=23, Z=24, X=0
                )

# one-hot 编码
def dna_onehot_outdateed(Xdna):
    listtmp = list()
    for index, row in Xdna.iterrows():
        row = [prot_dict[x] if x in prot_dict else x for x in row['seq']]
        listtmp.append(row)
    return pd.DataFrame(listtmp)

# one-hot 编码
def dna_onehot(Xdna):
    listtmp = []
    listtmp =Xdna.seq.parallel_apply(lambda x: np.array([prot_dict.get(item) for item in x]))
    listtmp = pd.DataFrame(np.stack(listtmp))
    listtmp = pd.concat( [Xdna.iloc[:,0:2], listtmp], axis=1)
    return listtmp


def lrmain(X_train_std, Y_train, X_test_std, Y_test, type='binary'):
    logreg = linear_model.LogisticRegression(
                                            solver = 'liblinear',
                                            multi_class='auto',
                                            verbose=False,
                                            max_iter=100
                                        )
    # sc = StandardScaler()
    # X_train_std = sc.fit_transform(X_train_std)
    logreg.fit(X_train_std, Y_train)
    predict = logreg.predict(X_test_std)
    lrpredpro = logreg.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, lrpredpro

def knnmain(X_train_std, Y_train, X_test_std, Y_test, type='binary'):
    knn=KNeighborsClassifier(n_neighbors=5, n_jobs=-2)
    knn.fit(X_train_std, Y_train)
    predict = knn.predict(X_test_std)
    lrpredpro = knn.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, lrpredpro

def svmmain(X_train_std, Y_train, X_test_std, Y_test):
    svcmodel = SVC(probability=True, kernel='rbf', tol=0.001)
    svcmodel.fit(X_train_std, Y_train.ravel(), sample_weight=None)
    predict = svcmodel.predict(X_test_std)
    predictprob =svcmodel.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def xgmain(X_train_std, Y_train, X_test_std, Y_test, type='binary'):

    x_train, x_vali, y_train, y_vali = train_test_split(X_train_std, Y_train, test_size=0.2, random_state=1)
    eval_set = [(x_train, y_train), (x_vali, y_vali)]

    if type=='binary':
        model = XGBClassifier(
                                objective='binary:logistic', 
                                random_state=15, 
                                use_label_encoder=False, 
                                n_jobs=-1, 
                                eval_metric='mlogloss',
                                min_child_weight=15, 
                                max_depth=15, 
                                n_estimators=300
                            )
        model.fit(x_train, y_train.ravel(), eval_metric="logloss", eval_set=eval_set, verbose=False)
    if type=='multi':
        model = XGBClassifier(
                        min_child_weight=6, 
                        max_depth=6, 
                        objective='multi:softmax', 
                        num_class=len(set(Y_train)), 
                        use_label_encoder=False,
                        n_estimators=120
                    )
        model.fit(x_train, y_train, eval_metric="mlogloss", eval_set=eval_set, verbose=False)
    
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def dtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def rfmain(X_train_std, Y_train, X_test_std, Y_test):
    model = RandomForestClassifier(oob_score=True, random_state=10, n_jobs=-2)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob

def gbdtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = GradientBoostingClassifier(random_state=10)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob
    

def caculateMetrix(groundtruth, predict, baselineName, type='binary'):
    acc = metrics.accuracy_score(groundtruth, predict)
    if type == 'binary':
        precision = metrics.precision_score(groundtruth, predict, zero_division=True )
        recall = metrics.recall_score(groundtruth, predict,  zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, zero_division=True)
        tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()
        npv = tn/(fn+tn+1.4E-45)
        print(baselineName, '\t\t%f' %acc,'\t%f'% precision,'\t\t%f'%npv,'\t%f'% recall,'\t%f'% f1, '\t', 'tp:',tp,'fp:',fp,'fn:',fn,'tn:',tn)
    
    if type == 'multi':
        precision = metrics.precision_score(groundtruth, predict, average='macro', zero_division=True )
        recall = metrics.recall_score(groundtruth, predict, average='macro', zero_division=True)
        f1 = metrics.f1_score(groundtruth, predict, average='macro', zero_division=True)
        print('%12s'%baselineName, ' \t\t%f '%acc,'\t%f'% precision, '\t\t%f'% recall,'\t%f'% f1)


def evaluate(baslineName, X_train_std, Y_train, X_test_std, Y_test, type='binary'):

    if baslineName == 'lr':
        groundtruth, predict, predictprob = lrmain (X_train_std, Y_train, X_test_std, Y_test, type=type)
    elif baslineName == 'svm':
        groundtruth, predict, predictprob = svmmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='xg':
        groundtruth, predict, predictprob = xgmain(X_train_std, Y_train, X_test_std, Y_test, type=type)
    elif baslineName =='dt':
        groundtruth, predict, predictprob = dtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='rf':
        groundtruth, predict, predictprob = rfmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='gbdt':
        groundtruth, predict, predictprob = gbdtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='knn':
        groundtruth, predict, predictprob = knnmain(X_train_std, Y_train, X_test_std, Y_test)
    else:
        print('Baseline Name Errror')

    caculateMetrix(groundtruth=groundtruth,predict=predict,baselineName=baslineName, type=type)

    # acc = metrics.accuracy_score(groundtruth, predict)
    # precision = metrics.precision_score(groundtruth, predict, zero_division=1 )
    # recall = metrics.recall_score(groundtruth, predict)
    # f1 = metrics.f1_score(groundtruth, predict)
    # auroc = metrics.roc_auc_score(groundtruth, predictprob)
    # auprc = metrics.average_precision_score(groundtruth, predictprob)
    # tn, fp, fn, tp = metrics.confusion_matrix(groundtruth, predict).ravel()

    # npv = tn/(fn+tn+1.4E-45)
    
    # print(baslineName, '\t\t%f' %acc,'\t%f'% precision,'\t\t%f'%npv,'\t%f'% recall,'\t%f'% f1, '\t%f'% auroc,'\t%f'% auprc, '\t', 'tp:',tp,'fp:',fp,'fn:',fn,'tn:',tn)

def run_baseline(X_train, Y_train, X_test, Y_test, type='binary'):
    methods=['knn','lr', 'xg', 'dt', 'rf', 'gbdt']
    if type == 'binary':
        print('baslineName', '\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', '\t\t confusion Matrix')
    if type =='multi':
        print('%12s'%'baslineName', '\t\t', 'accuracy','\t', 'precision-macro \t', 'recall-macro','\t', 'f1-macro')
    for method in methods:
        evaluate(method, X_train, Y_train, X_test, Y_test, type=type)
        
    
    
def static_interval(data, span):
    """[summary]

    Args:
        data ([dataframe]): [需要统计的数据]
        span ([int]): [统计的间隔]

    Returns:
        [type]: [完成的统计列表]]
    """
    res = []
    count = 0
    for i in range(int(len(data)/span) + 1):
        lable = str(i*span) + '-' + str((i+1)* span -1 )
        num = data[(data.length>=(i*span)) & (data.length<(i+1)*span)]['count'].sum()
        res += [[lable, num]]
    return res
        

        
def getblast(train, test):
    
    table2fasta(train, '/tmp/train.fasta')
    table2fasta(test, '/tmp/test.fasta')
    
    cmd1 = r'diamond makedb --in /tmp/train.fasta -d /tmp/train.dmnd'
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  /tmp/test.fasta -o /tmp/test_fasta_results.tsv -b5 -c1 -k 1'
    cmd3 = r'rm -rf /tmp/*.fasta /tmp/*.dmnd /tmp/*.tsv'
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv('/tmp/test_fasta_results.tsv', sep='\t', names=['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore'])
    os.system(cmd3)
    return res_data

def getblast_fasta(trainfasta, testfasta):
    
    cmd1 = r'diamond makedb --in {0} -d /tmp/train.dmnd --quiet'.format(trainfasta)
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  {0} -o /tmp/test_fasta_results.tsv -b8 -c1 -k 1 --quiet'.format(testfasta)
    cmd3 = r'rm -rf /tmp/*.fasta /tmp/*.dmnd /tmp/*.tsv'
    # print(cmd1)
    os.system(cmd1)
    # print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv('/tmp/test_fasta_results.tsv', sep='\t', names=['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore'])
    os.system(cmd3)
    return res_data




def caculateMetrix_1(baselineName,tp, fp, tn,fn):
    sampleNum = tp+fp+tn+fn
    accuracy = (tp+tn)/sampleNum
    precision = tp/(tp+fp)
    npv = tn/(tn+fn)
    recall = tp/(tp+fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('{0} \t {1:.6f}  \t{2:.6f} \t\t {3:.6f}  \t{4:.6f}\t {5:.6f}\t\t \t \t \t tp:{6}  fp:{7}  fn:{8}  tn:{9}'.format(baselineName,accuracy, precision, npv, recall, f1, tp,fp,fn,tn))



def get_integrated_results(res_data, train, test, baslineName):
    # 给比对结果添加标签
    isEmzyme_dict = {v: k for k,v in zip(train.isemzyme, train.id )} 
    res_data['diamoion_pred'] = res_data['sseqid'].apply(lambda x: isEmzyme_dict.get(x))

    blast_res = pd.DataFrame
    blast_res = res_data[['id','pident','bitscore', 'diamoion_pred']]
    
    X_train = train.iloc[:,12:]
    X_test = test.iloc[:,12:]
    Y_train = train.iloc[:,2].astype('int')
    Y_test = test.iloc[:,2].astype('int')
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train).flatten()
    Y_test = np.array(Y_test).flatten()
    
    if baslineName == 'lr':
        groundtruth, predict, predictprob = lrmain (X_train, Y_train, X_test, Y_test)
    elif baslineName == 'svm':
        groundtruth, predict, predictprob = svmmain(X_train, Y_train, X_test, Y_test)
    elif baslineName =='xg':
        groundtruth, predict, predictprob = xgmain(X_train, Y_train, X_test, Y_test)
    elif baslineName =='dt':
        groundtruth, predict, predictprob = dtmain(X_train, Y_train, X_test, Y_test)
    elif baslineName =='rf':
        groundtruth, predict, predictprob = rfmain(X_train, Y_train, X_test, Y_test)
    elif baslineName =='gbdt':
        groundtruth, predict, predictprob = gbdtmain(X_train, Y_train, X_test, Y_test)
    else:
        print('Baseline Name Errror')
    
    test_res = pd.DataFrame()
    test_res[['id', 'name','isemzyme','ec_number']] = test[['id','name','isemzyme','ec_number']]
    test_res.reset_index(drop=True, inplace=True)

    #拼合比对结果到测试集
    test_merge_res = pd.merge(test_res, blast_res, on='id', how='left')
    test_merge_res['xg_pred'] = predict
    test_merge_res['xg_pred_prob'] = predictprob
    test_merge_res['groundtruth'] = groundtruth
    
    test_merge_res['final_pred'] = ''
    for index, row in test_merge_res.iterrows():
        if (row.diamoion_pred == True) | (row.diamoion_pred == False):
            with pd.option_context('mode.chained_assignment', None):
                test_merge_res['final_pred'][index] = row.diamoion_pred
        else:
            with pd.option_context('mode.chained_assignment', None):
                test_merge_res['final_pred'][index] = row.xg_pred

    # 计算指标
    tp = len(test_merge_res[test_merge_res.groundtruth & test_merge_res.final_pred])
    fp = len(test_merge_res[(test_merge_res.groundtruth ==False) & (test_merge_res.final_pred)])
    tn = len(test_merge_res[(test_merge_res.groundtruth ==False) & (test_merge_res.final_pred ==False)])
    fn = len(test_merge_res[(test_merge_res.groundtruth ) & (test_merge_res.final_pred == False)])
    caculateMetrix_1(baslineName,tp, fp, tn,fn)


def run_integrated(res_data, train, test):
    methods=['lr','xg', 'dt', 'rf', 'gbdt']
    print('baslineName', '\t\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', 'auroc','\t\t', 'auprc', '\t\t confusion Matrix')
    for method in methods:
        get_integrated_results(res_data, train, test, method)



#region 将多功能的EC编号展开，返回唯一的EC编号列表
def get_distinct_ec(ecnumbers):
    """
    将多功能的EC编号展开，返回唯一的EC编号列表
    Args:
        ecnumbers: EC_number 列

    Returns: 排序好的唯一EC列表

    """
    result_list=[]
    for item in ecnumbers:
        ecarray = item.split(',')
        for subitem in ecarray:
            result_list+=[subitem.strip()]
    return sorted(list(set(result_list)))
#endregion



#region 将多功能酶拆解为多个单功能酶
def split_ecdf_to_single_lines(full_table):
    """
    将多功能酶拆解为多个单功能酶
    Args:
        full_table: 包含EC号的完整列表
    并 1.去除酶号前后空格
    并 2. 将酶号拓展为4位的标准格式
    Returns: 展开后的EC列表，每个EC号一行
    """
    resDf = pd.DataFrame(columns=full_table.columns.values)
    for index, row in tqdm(full_table.iterrows()):
        if row.ec_number.strip()=='-':   #若是非酶直接返回
            row.ec_number='-'
            row.ec_number = row.ec_number.strip()
            resDf = resDf.append(row, ignore_index=True)
        else:
            ecs = row.ec_number.split(',') #拆解多功能酶
            for ec in ecs:
                ec = ec.strip()
                ecarray=ec.split('.') #拆解每一位
                if ecarray[3] == '':  #若是最后一位是空，补足_
                    ec=ec+'-'
                row.ec_number = ec.strip()
                resDf = resDf.append(row, ignore_index=True)
    return resDf

#endregion


