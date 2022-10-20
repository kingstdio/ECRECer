import pandas as pd
import numpy as np
import os,string, random
from datetime import datetime
import config as cfg


#region 准备Slice使用的文件
def prepare_slice_file(x_data, y_data, x_file, y_file, ec_label_dict):
    """
    准备Slice使用的文件
    Args:
        x_data: X数据
        y_data: Y数据
        x_file: X文件路径
        y_file: Y文件路径
        ec_label_dict: EC转标签字典

    Returns:

    """
    if (os.path.exists(x_file) == False) or (cfg.UPDATE_MODEL ==True):
         to_file_matrix(file=x_file, ds=x_data.round(cfg.SAMPLING_BIT), col_num=cfg.FEATURE_NUM, stype='feature')

    if (os.path.exists(y_file) == False) or (cfg.UPDATE_MODEL ==True):
        with pd.option_context('mode.chained_assignment', None): 
            y_data['tags'] = 1
        to_file_matrix(
            file=y_file,
            ds=y_data,
            col_num=max(ec_label_dict.values()),
            stype='label'
        )

    print('slice files prepared success')

def prepare_slice_file_onlyx(x_data, x_file):
    """
    准备Slice使用的文件
    Args:
        x_data: X数据
        x_file: X文件路径

    Returns:

    """
    if (os.path.exists(x_file) == False) or (cfg.UPDATE_MODEL ==True):
         to_file_matrix(file=x_file, ds=x_data.round(cfg.SAMPLING_BIT), col_num=cfg.FEATURE_NUM, stype='feature')

    print('slice files prepared success')   
#endregion

#region 创建slice需要的数据文件
def to_file_matrix(file, ds, col_num, stype='label'):
    """[创建slice需要的数据文件]
    Args:
        file ([string]): [要保存的文件名]
        ds ([DataFrame]): [数据]
        col_num ([int]): [有多少列]
        stype (str, optional): [文件类型：feature，label]. Defaults to 'label'.
    """
    if os.path.exists(file) & (cfg.UPDATE_MODEL ==False):
        return 'file exist'
    
    if stype== 'label':
        seps = ':'
    if stype == 'feature':
        seps = ' '
    ds.to_csv(file, index= 0, header =0 , sep= seps)

    cmd ='''sed -i '1i {0} {1}' {2}'''.format(len(ds), col_num, file)
    os.system(cmd)
#endregion

# region 需要写fasta的dataFame格式
def save_table2fasta(dataset, file_out):
    """[summary]
    Args:
        dataset ([DataFrame]): 需要写fasta的dataFame格式[id, seq]
        file_out ([type]): [description]
    """
    if ~os.path.exists(file_out):
        file = open(file_out, 'w')
        for index, row in dataset.iterrows():
            file.write('>{0}\n'.format(row['id']))
            file.write('{0}\n'.format(row['seq']))
        file.close()
    print('Write finished')


# endregion

# region 获取序列比对结果
def getblast(ref_fasta, query_fasta, results_file):
    """[获取序列比对结果]
    Args:
        ref_fasta ([string]]): [训练集fasta文件]
        query_fasta ([string]): [测试数据集fasta文件]
        results_file ([string]): [要保存的结果文件]

    Returns:
        [DataFrame]: [比对结果]
    """

    if os.path.exists(results_file):
        res_data = pd.read_csv(results_file, sep='\t', names=cfg.BLAST_TABLE_HEAD)
        return res_data


    cmd1 = r'diamond makedb --in {0} -d /tmp/train.dmnd'.format(ref_fasta)
    cmd2 = r'diamond blastp -d /tmp/train.dmnd  -q  {0} -o {1} -b8 -c1 -k 1'.format(query_fasta, results_file)
    cmd3 = r'rm -rf /tmp/*.dmnd'

    print(cmd1)
    os.system(cmd1)
    print(cmd2)
    os.system(cmd2)
    res_data = pd.read_csv(results_file, sep='\t', names=cfg.BLAST_TABLE_HEAD)
    os.system(cmd3)
    return res_data


# endregion

#region 创建diamond数据库
def make_diamond_db(dbtable, to_db_file):
    """[创建diamond数据库]

    Args:
        dbtable ([DataTable]): [[ID，SEQ]DataFame]
        to_db_file ([string]): [数据库文件]
    """
    save_table2fasta(dataset=dbtable, file_out=cfg.TEMPDIR+'dbtable.fasta')
    cmd = r'diamond makedb --in {0} -d {1}'.format(cfg.TEMPDIR+'dbtable.fasta', to_db_file)
    os.system(cmd)
    cmd = r'rm -rf {0}'.format(cfg.TEMPDIR+'dbtable.fasta')
    os.system(cmd)
#endregion 

#region 为blast序列比对添加结果标签
def blast_add_label(blast_df, trainset,):
    """[为blast序列比对添加结果标签]

    Args:
        blast_df ([DataFrame]): [序列比对结果]
        trainset ([DataFrame]): [训练数据]

    Returns:
        [Dataframe]: [添加标签后的数据]
    """
    res_df = blast_df.merge(trainset, left_on='sseqid', right_on='id', how='left')
    res_df = res_df.rename(columns={'id_x': 'id',
                                              'isemzyme': 'isenzyme_blast',
                                              'functionCounts': 'functionCounts_blast',
                                              'ec_number': 'ec_blast',
                                              'ec_specific_level': 'ec_specific_level_blast'
                                              })
    return res_df.iloc[:,np.r_[0,13:16]]
#endregion

#region loading embedding features
def load_data_embedding(embedding_type):
    """loading embedding features

    Args:
        embedding_type (string): one of ['one-hot', 'unirep', 'esm0', 'esm32', 'esm33']

    Returns:
        DataFrame: features
    """

    if embedding_type=='one-hot':   #one-hot
        feature = pd.read_feather(cfg.FILE_FEATURE_ONEHOT)

    if embedding_type=='unirep':   #unirep
        feature = pd.read_feather(cfg.FILE_FEATURE_UNIREP)

    if embedding_type=='esm0':   #esm0
        feature = pd.read_feather(cfg.FILE_FEATURE_ESM0)
    
    if embedding_type =='esm32':  #esm32
        feature = pd.read_feather(cfg.FILE_FEATURE_ESM32)
    
    if embedding_type =='esm33':  #esm33
        feature = pd.read_feather(cfg.FILE_FEATURE_ESM33)

    return feature
#endregion



def get_blast_prediction(reference_db, train_frame, test_frame, results_file, identity_thres=0.2):

    save_table2fasta(dataset=test_frame.iloc[:,np.r_[0,5]], file_out=cfg.TEMPDIR+'test.fasta')
    cmd = r'diamond blastp -d {0}  -q  {1} -o {2} -b8 -c1 -k 1'.format(reference_db, cfg.TEMPDIR+'test.fasta', results_file)
    print(cmd)
    os.system(cmd)
    res_data = pd.read_csv(results_file, sep='\t', names=cfg.BLAST_TABLE_HEAD)
    res_data = res_data[res_data.pident >= identity_thres] # 按显著性阈值过滤
    res_df = res_data.merge(train_frame, left_on='sseqid', right_on='id', how='left')
    res_df = res_df.rename(columns={'id_x': 'id',
                                            'isemzyme': 'isemzyme_blast',
                                            'functionCounts': 'functionCounts_blast',
                                            'ec_number': 'ec_number_blast',
                                            'ec_specific_level': 'ec_specific_level_blast'
                                            })

    return res_df.iloc[:,np.r_[0,2,13:17]]
    
#region 读取cdhit聚类结果
def get_cdhit_results(cdhit_clstr_file):
    """读取cdhit聚类结果

    Args:
        cdhit_clstr_file (string): 聚类结果文件

    Returns:
        DataFrame: ['cluster_id','uniprot_id','identity'， 'is_representative']
    """
    counter = 0
    res = []
    with open(cdhit_clstr_file,'r') as f:
        for line in f:
            if 'Cluster' in line:
                cluster_id = line.replace('>Cluster','').replace('\n', '').strip()
                continue
            str_uids= line.replace('\n','').split('>')[1].replace('at ','').split('... ')
                        
            if '*' in str_uids[1]:
                identity = 1
                isrep = True
            else:
                identity = float(str_uids[1].strip('%')) /100
                isrep = False

            res = res +[[cluster_id, str_uids[0], identity, isrep ]]

    resdf = pd.DataFrame(res, columns=['cluster_id','uniprot_id','identity', 'is_representative']) #转换为DataFrame
    return resdf
#endregion

def pycdhit(uniportid_seq_df, identity=0.4, thred_num=4):
    """CD-HIT 序列聚类

    Args:
        uniportid_seq_df (DataFrame): [uniprot_id, seq] 蛋白DataFrame
        identity (float, optional): 聚类阈值. Defaults to 0.4.
        thred_num (int, optional): 聚类线程数. Defaults to 4.

    Returns:
        聚类结果 DataFrame: [cluster_id,uniprot_id,identity,is_representative,cluster_size]
    """
    if identity>=0.7:
        word_size = 5
    elif identity>=0.6:
        word_size = 4
    elif identity >=0.5:
        word_size = 3
    elif identity >=0.4:
        word_size =2
    else:
        word_size = 5

    # 定义输入输出文件名


    
    time_stamp_str = datetime.now().strftime("%Y-%m-%d_%H_%M_%S_")+''.join(random.sample(string.ascii_letters + string.digits, 16))
    cd_hit_fasta = f'{cfg.TEMPDIR}cdhit_test_{time_stamp_str}.fasta'
    cd_hit_results = f'{cfg.TEMPDIR}cdhit_results_{time_stamp_str}'
    cd_hit_cluster_res_file =f'{cfg.TEMPDIR}cdhit_results_{time_stamp_str}.clstr'

    # 写聚类fasta文件
    save_table2fasta(uniportid_seq_df, cd_hit_fasta)

    # cd-hit聚类
    cmd = f'cd-hit -i {cd_hit_fasta} -o {cd_hit_results} -c {identity} -n {word_size} -T {thred_num} -M 0 -g 1 -sc 1 -sf 1 > /dev/null 2>&1'
    os.system(cmd)
    cdhit_cluster = get_cdhit_results(cdhit_clstr_file=cd_hit_cluster_res_file)

    cluster_size = cdhit_cluster.cluster_id.value_counts()
    cluster_size = pd.DataFrame({'cluster_id':cluster_size.index,'cluster_size':cluster_size.values})
    cdhit_cluster = cdhit_cluster.merge(cluster_size, on='cluster_id', how='left')
    
    cmd = f'rm -f {cd_hit_fasta} {cd_hit_results} {cd_hit_cluster_res_file}'
    os.system(cmd)

    return cdhit_cluster

#region 打印模型的重要指标，排名topN指标
def importance_features_top(model, x_train, topN=10):
    """[打印模型的重要指标，排名topN指标]
    Args:
        model ([type]): [description]
        x_train ([type]): [description]
        topN (int, optional): [description]. Defaults to 10.
    """
    print("打印XGBoost重要指标")
    feature_importances_ = model.feature_importances_
    feature_names = x_train.columns
    importance_col = pd.DataFrame([*zip(feature_names, feature_importances_)],  columns=['features', 'weight'])
    importance_col_desc = importance_col.sort_values(by='weight', ascending=False)
    print(importance_col_desc.iloc[:topN, :])
#endregion

#test

if __name__ =='__main__':
    print('success')