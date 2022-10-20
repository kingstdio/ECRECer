from Bio import SeqIO
import gzip
import re
from tqdm import tqdm
import time
import sys,os
sys.path.append(os.getcwd())
import config as cfg
import pandas as pd

#region 从gizp读取数据
def read_file_from_gzip(file_in_path, file_out_path, extract_type, save_file_type='tsv'):
    """从原始Zip file 中解析数据

    Args:
        file_in_path ([string]): [输入文件路径]
        file_out_path ([string]): [输出文件路径]]
        extract_type ([string]): [抽取数据的类型：with_ec, without_ec, full]]
    """
    if save_file_type == 'feather':
        outpath = file_out_path
        file_out_path = cfg.TEMPDIR +'temprecords.tsv'

    table_head = [  'id', 
                    'name',
                    'isenzyme',
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
    counter = 0
    saver = 0 
    file_write_obj = open(file_out_path,'w')
    with gzip.open(file_in_path, "rt") as handle:
        file_write_obj.writelines('\t'.join(table_head))
        file_write_obj.writelines('\n')
        for record in tqdm( SeqIO.parse(handle, 'swiss'), position=1, leave=True):
            res = process_record(record, extract_type= extract_type)
            counter+=1
            if counter %10==0:
                file_write_obj.flush()
            # if saver%100000==0:
            #     print(saver)
            if len(res) >0:
                saver +=1
                file_write_obj.writelines('\t'.join(map(str,res)))
                file_write_obj.writelines('\n')
            else:
                continue
    file_write_obj.close()

    if save_file_type == 'feather':
        indata = pd.read_csv(cfg.TEMPDIR +'temprecords.tsv', sep='\t')
        indata.to_feather(outpath)

 
 #endregion

#region 提取单条含有EC号的数据
def process_record(record, extract_type='with_ec'):
    """
    提取单条含有EC号的数据
    Args:
        record ([type]): uniprot 中的记录节点
        extract_type (string, optional): 提取的类型，可选值：with_ec, without_ec, full。默认为有EC号（with_ec).
    Returns:
        [type]: [description]
    """


    description = record.description
    isEnzyme = 'EC=' in description     #有EC号的被认为是酶，否则认为是酶
    isMultiFunctional = False
    functionCounts = 0
    ec_specific_level =0


    if isEnzyme:
        ec = str(re.findall(r"EC=[0-9,.\-;]*",description)
                 ).replace('EC=','').replace('\'','').replace(']','').replace('[','').replace(';','').strip()

        #统计酶的功能数
        isMultiFunctional = ',' in ec
        functionCounts = ec.count(',') + 1


        # - 单功能酶
        if not isMultiFunctional:
            levelCount = ec.count('-')
            ec_specific_level = 4-levelCount

        else: # -多功能酶
            ecarray = ec.split(',')
            for subec in ecarray:
                current_ec_level = 4- subec.count('-')
                if ec_specific_level < current_ec_level:
                    ec_specific_level = current_ec_level
    else:
        ec = '-'
    
    id = record.id.strip()
    name = record.name.strip()
    seq = record.seq.strip()
    date_integrated = record.annotations.get('date').strip()
    date_sequence_update = record.annotations.get('date_last_sequence_update').strip()
    date_annotation_update = record.annotations.get('date_last_annotation_update').strip()
    seqlength = len(seq)
    res = [id, name, isEnzyme, isMultiFunctional, functionCounts, ec,ec_specific_level, date_integrated, date_sequence_update, date_annotation_update,  seq, seqlength]

    if extract_type == 'full':
        return res

    if extract_type == 'with_ec':
        if isEnzyme:
            return res
        else:
            return []

    if extract_type == 'without_ec':
        if isEnzyme:
            return []
        else:
            return res
#endregion

def run_exact_task(infile, outfile):
    start =  time.process_time()
    extract_type ='full'
    read_file_from_gzip(file_in_path=infile, file_out_path=outfile, extract_type=extract_type)
    end =  time.process_time()
    print('finished use time %6.3f s' % (end - start))

if __name__ =="__main__":
    start =  time.process_time()
    in_filepath_sprot = cfg.FILE_LATEST_SPROT
    out_filepath_sprot = cfg.FILE_LATEST_SPROT_FEATHER
    
    in_filepath_trembl = cfg.FILE_LATEST_TREMBL
    out_filepath_trembl = cfg.FILE_LATEST_TREMBL_FEATHER

    extract_type ='full'


    # read_file_from_gzip(file_in_path=in_filepath_sprot, file_out_path=out_filepath_sprot, extract_type=extract_type, save_file_type='feather')

    read_file_from_gzip(file_in_path=in_filepath_trembl, file_out_path=out_filepath_trembl, extract_type=extract_type, save_file_type='feather')
    end =  time.process_time()
    print('finished use time %6.3f s' % (end - start))

   