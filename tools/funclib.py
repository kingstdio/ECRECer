from contextlib import nullcontext
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn import linear_model, datasets
from sklearn.svm import SVC
from sklearn import tree
from tkinter import _flatten
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
from Bio import SeqIO
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
                                            solver = 'saga',
                                            multi_class='auto',
                                            verbose=False,
                                            n_jobs=-1,
                                            max_iter=10000
                                        )
    # sc = StandardScaler()
    # X_train_std = sc.fit_transform(X_train_std)
    logreg.fit(X_train_std, Y_train)
    predict = logreg.predict(X_test_std)
    lrpredpro = logreg.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, lrpredpro, logreg

def knnmain(X_train_std, Y_train, X_test_std, Y_test, type='binary'):
    knn=KNeighborsClassifier(n_neighbors=5, n_jobs=16)
    knn.fit(X_train_std, Y_train)
    predict = knn.predict(X_test_std)
    lrpredpro = knn.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, lrpredpro, knn

def svmmain(X_train_std, Y_train, X_test_std, Y_test):
    svcmodel = SVC(probability=True, kernel='rbf', tol=0.001)
    svcmodel.fit(X_train_std, Y_train.ravel(), sample_weight=None)
    predict = svcmodel.predict(X_test_std)
    predictprob =svcmodel.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob,svcmodel

def xgmain(X_train_std, Y_train, X_test_std, Y_test, type='binary', vali=True):

    x_train, x_vali, y_train, y_vali = train_test_split(X_train_std, Y_train, test_size=0.2, random_state=1)
    eval_set = [(x_train, y_train), (x_vali, y_vali)]

    if type=='binary':
        # model = XGBClassifier(
        #                         objective='binary:logistic', 
        #                         random_state=15, 
        #                         use_label_encoder=False, 
        #                         n_jobs=-1, 
        #                         eval_metric='mlogloss',
        #                         min_child_weight=15, 
        #                         max_depth=15, 
        #                         n_estimators=300,
        #                         tree_method = 'gpu_hist',
        #                         learning_rate = 0.01
        #                     )
        model = XGBClassifier(
                                objective='binary:logistic', 
                                random_state=13, 
                                use_label_encoder=False, 
                                n_jobs=-2, 
                                eval_metric='auc',
                                max_depth=6,
                                n_estimators= 300,
                                tree_method = 'gpu_hist',
                                learning_rate = 0.01,
                                gpu_id=0
                            )
        model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
        # model.fit(t1_x_train, t1_y_train,  eval_set=t1_eval_set, verbose=100)
    if type=='multi':
        model = XGBClassifier(
                        min_child_weight=6, 
                        max_depth=6, 
                        objective='multi:softmax', 
                        num_class=len(set(Y_train)), 
                        use_label_encoder=False,
                        n_estimators=120
                    )
        if vali:
            model.fit(x_train, y_train, eval_metric="mlogloss", eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train_std, Y_train, eval_metric="mlogloss", eval_set=None, verbose=False)
    
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob, model

def dtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = tree.DecisionTreeClassifier()
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob,model

def rfmain(X_train_std, Y_train, X_test_std, Y_test):
    model = RandomForestClassifier(oob_score=True, random_state=10, n_jobs=-2)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob,model

def gbdtmain(X_train_std, Y_train, X_test_std, Y_test):
    model = GradientBoostingClassifier(random_state=10)
    model.fit(X_train_std, Y_train.ravel())
    predict = model.predict(X_test_std)
    predictprob = model.predict_proba(X_test_std)
    groundtruth = Y_test
    return groundtruth, predict, predictprob, model
    

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
        groundtruth, predict, predictprob, model = lrmain (X_train_std, Y_train, X_test_std, Y_test, type=type)
    elif baslineName == 'svm':
        groundtruth, predict, predictprob, model = svmmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='xg':
        groundtruth, predict, predictprob, model = xgmain(X_train_std, Y_train, X_test_std, Y_test, type=type)
    elif baslineName =='dt':
        groundtruth, predict, predictprob, model = dtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='rf':
        groundtruth, predict, predictprob, model = rfmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='gbdt':
        groundtruth, predict, predictprob, model = gbdtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='knn':
        groundtruth, predict, predictprob, model = knnmain(X_train_std, Y_train, X_test_std, Y_test, type=type)
    else:
        print('Baseline Name Errror')

    caculateMetrix(groundtruth=groundtruth,predict=predict,baselineName=baslineName, type=type)

def evaluate_2(baslineName, X_train_std, Y_train, X_test_std, Y_test, type='binary'):

    if baslineName == 'lr':
        groundtruth, predict, predictprob, model = lrmain (X_train_std, Y_train, X_test_std, Y_test, type=type)
    elif baslineName == 'svm':
        groundtruth, predict, predictprob, model = svmmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='xg':
        groundtruth, predict, predictprob, model = xgmain(X_train_std, Y_train, X_test_std, Y_test, type=type, vali=False)
    elif baslineName =='dt':
        groundtruth, predict, predictprob, model = dtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='rf':
        groundtruth, predict, predictprob, model = rfmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='gbdt':
        groundtruth, predict, predictprob, model = gbdtmain(X_train_std, Y_train, X_test_std, Y_test)
    elif baslineName =='knn':
        groundtruth, predict, predictprob, model = knnmain(X_train_std, Y_train, X_test_std, Y_test, type=type)
    else:
        print('Baseline Name Errror')

    caculateMetrix(groundtruth=groundtruth,predict=predict,baselineName=baslineName, type=type)

def run_baseline(X_train, Y_train, X_test, Y_test, type='binary'):
    methods=['knn','lr', 'xg', 'dt', 'rf', 'gbdt']
    if type == 'binary':
        print('baslineName', '\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', '\t\t confusion Matrix')
    if type =='multi':
        print('%12s'%'baslineName', '\t\t', 'accuracy','\t', 'precision-macro \t', 'recall-macro','\t', 'f1-macro')
    for method in methods:
        evaluate(method, X_train, Y_train, X_test, Y_test, type=type)
 
def run_baseline_2(X_train, Y_train, X_test, Y_test, type='binary'):
    methods=['knn', 'xg', 'dt', 'rf', 'gbdt']
    if type == 'binary':
        print('baslineName', '\t', 'accuracy','\t', 'precision(PPV) \t NPV \t\t', 'recall','\t', 'f1', '\t\t', '\t\t confusion Matrix')
    if type =='multi':
        print('%12s'%'baslineName', '\t\t', 'accuracy','\t', 'precision-macro \t', 'recall-macro','\t', 'f1-macro')
    for method in methods:
        evaluate_2(method, X_train, Y_train, X_test, Y_test, type=type)
    
    
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
    
    cmd1 = r'diamond makedb --in /tmp/train.fasta -d /tmp/train.dmnd --quiet'
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  /tmp/test.fasta -o /tmp/test_fasta_results.tsv -b5 -c1 -k 1 --quiet'
    cmd3 = r'rm -rf /tmp/*.fasta /tmp/*.dmnd /tmp/*.tsv'
    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv('/tmp/test_fasta_results.tsv', sep='\t', names=['id', 'sseqid', 'pident', 'length','mismatch','gapopen','qstart','qend','sstart','send','evalue','bitscore'])
    os.system(cmd3)
    return res_data

def getblast_usedb(db, test):
    table2fasta(test, '/tmp/test.fasta')
    cmd2 = r'diamond blastp -d {0}  -q  /tmp/test.fasta -o /tmp/test_fasta_results.tsv -b5 -c1 -k 1 --quiet'.format(db)
    cmd3 = r'rm -rf /tmp/*.fasta /tmp/*.dmnd /tmp/*.tsv'

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


#region 统计EC数
def stiatistic_ec_num(eclist):
    """统计EC数量

    Args:
        eclist (list): 可包含多功能酶的EC列表，用；分割

    Returns:
        int: 列表中包含的独立EC数量
    """
    eclist = list(eclist.flatten()) #展成1维
    eclist = _flatten([item.split(';') for item in eclist]) #分割多功能酶
    eclist = [item.strip() for item in eclist] # 去空格
    num_ecs = len(set(eclist))
    return num_ecs
#endregion

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
    listres=full_table.parallel_apply(lambda x: split_ecdf_to_single_lines_pr_record(x)  , axis=1)
    temp_li = []
    for res in tqdm(listres):
        for j in res:
            temp_li = temp_li + [j]
    resDf = pd.DataFrame(temp_li,columns=full_table.columns.values)

    return resDf
#endregion


def split_ecdf_to_single_lines_pr_record(row):
    resDf = []

    if row.ec_number.strip()=='-':   #若是非酶直接返回
            row.ec_number='-'
            row.ec_number = row.ec_number.strip()
            resDf = row.values
            return [[row.id, row.seq, row.ec_number]]
    else:
        ecs = row.ec_number.split(',') #拆解多功能酶
        if len(ecs) ==1:               # 单功能酶直接返回
            return [[row.id, row.seq, row.ec_number]]
        for ec in ecs:
            ec = ec.strip()
            ecarray=ec.split('.') #拆解每一位
            if ecarray[3] == '':  #若是最后一位是空，补足_
                ec=ec+'-'
            row.ec_number = ec.strip()
            resDf = resDf + [[row.id, row.seq, ec]]
    return  resDf



def load_fasta_to_table(file):
    """[Load fasta file to DataFrame]

    Args:
        file ([string]): [fasta file location]

    Returns:
        [DataFrame]: [loaded fasta in DF format]
    """
    if os.path.exists(file) == False:
        print('file not found:{0}'.format(file))
        return nullcontext

    input_data=[]
    for record in SeqIO.parse(file, format='fasta'):
        input_data=input_data +[[record.id, str(record.seq)]]
    input_df = pd.DataFrame(input_data, columns=['id','seq'])
    return input_df

def load_deepec_resluts(filepath):
    """load deepec predicted resluts

    Args:
        filepath (string): deepec predicted file

    Returns:
        DataFrame: columns=['id', 'ec_deepec']
    """
    res_deepec = pd.read_csv(f'{filepath}', sep='\t',names=['id', 'ec_number'], header=0 )
    res_deepec.ec_number=res_deepec.apply(lambda x: x['ec_number'].replace('EC:',''), axis=1)
    res_deepec.columns = ['id','ec_deepec']
    res = []
    for index, group in  res_deepec.groupby('id'):
        if len(group)==1:
            res = res + [[group.id.values[0], group.ec_deepec.values[0]]]
        else:
            ecs_str = ','.join(group.ec_deepec.values)
            res = res +[[group.id.values[0],ecs_str]] 
    res_deepec = pd.DataFrame(res, columns=['id', 'ec_deepec'])
    return res_deepec


#region
def load_praim_res(resfile):
    """[加载PRIAM的预测结果]
    Args:
        resfile ([string]): [结果文件]
    Returns:
        [DataFrame]: [结果]
    """
    f = open(resfile)
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

def load_catfam_res(resfile):
    res_catfam = pd.read_csv(resfile, sep='\t', names=['id', 'ec_catfam'])
    return res_catfam


def load_ecpred_res(resfile):
    res_ecpred = pd.read_csv(f'{resfile}', sep='\t', header=0)
    res_ecpred = res_ecpred.rename(columns={'Protein ID':'id','EC Number':'ec_ecpred','Confidence Score(max 1.0)':'pident_ecpred'})
    res_ecpred['ec_ecpred']= res_ecpred.ec_ecpred.apply(lambda x : '-' if x=='non Enzyme' else x) 
    return res_ecpred