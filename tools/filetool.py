'''
Author: Zhenkun Shi
Date: 2022-10-05 05:24:14
LastEditors: Zhenkun Shi
LastEditTime: 2022-10-05 05:25:15
FilePath: /DMLF/tools/filetool.py
Description: 

Copyright (c) 2022 by tibd, All Rights Reserved. 
'''

from retrying import retry
from matplotlib.pyplot import axis
import urllib3
import pandas as pd
import os
import sys
from shutil import copyfile
from sys import exit
import zipfile


#region 下载文件
@retry(stop_max_attempt_number=10, wait_random_min=10, wait_random_max=20)
def download(download_url, save_file):
    """[从网址下载文件]
    Args:
        download_url ([Stirng]): [下载路径]
        save_file ([String]): [保存文件路径]
    """
    http = urllib3.PoolManager()
    response = http.request('GET', download_url)
    with open(save_file, 'wb') as f:
        f.write(response.data)
    response.release_conn()
#endregion


def wget(download_url, save_file, verbos=False):
    process = os.popen('which wget') # return file
    output = process.read()
    if output =='':
        print('wget not installed')
    else:
        if verbos:
            cmd = 'wget ' + download_url + ' -O ' + save_file
        else:
            cmd = 'wget -q ' + download_url + ' -O ' + save_file
        print (cmd)
        process = os.popen(cmd)
        output = process.read()
    process.close()

def convert_DF_dateTime(inputdf):
    """[Covert unisprot csv records datatime]

    Args:
        inputdf ([DataFrame]): [input dataFrame]

    Returns:
        [DataFrame]: [converted DataFrame]
    """
    inputdf.date_integraged = pd.to_datetime(inputdf['date_integraged'])
    inputdf.date_sequence_update = pd.to_datetime(inputdf['date_sequence_update'])
    inputdf.date_annotation_update = pd.to_datetime(inputdf['date_annotation_update'])
    inputdf = inputdf.sort_values(by='date_integraged', ascending=True)
    inputdf.reset_index(drop=True, inplace=True)
    return inputdf

def get_file_names_in_dir(dataroot, filetype='all'):
    """返回某个文件夹下的文件列表，可以指定文件类型

    Args:
        dataroot (string): 文件夹目录
        filetype (str, optional): 文件类型. Defaults to ''.

    Returns:
        DataFrame: columns=['filename','filetype','filename_no_suffix']
    """
    exist_file_df = pd.DataFrame(os.listdir(dataroot), columns=['filename'])

    if len(exist_file_df)!=0:
        exist_file_df['filetype'] = exist_file_df.filename.apply(lambda x: x.split('.')[-1])
        exist_file_df['filename_no_suffix'] = exist_file_df.apply(lambda x: x['filename'].replace(('.'+str(x['filetype']).strip()), ''), axis=1)
        if filetype !='all':
            exist_file_df = exist_file_df[exist_file_df.filetype==filetype]
        return exist_file_df
    else:
        return pd.DataFrame(columns=['filename','filetype','filename_no_suffix'])


def copy(source, target):
    """ 拷贝文件

    Args:
        source (string): source
        target (string): target
    """
    try:
        copyfile(src=source, dst=target)
    except IOError as e:
        print("Unable to copy file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)

def delete(filepath):
    """删除文件

    Args:
        filepath (string): 文件全路径
    """
    try:
        os.remove(filepath)
    except IOError as e:
        print("Unable to delete file. %s" % e)
        exit(1)
    except:
        print("Unexpected error:", sys.exc_info())
        exit(1)

#region unzip file
def unzipfile(filename, target_dir):
    """uzip file

    Args:
        zipfile (string): zip file full path
        target_dir (string): target dir
    """
    with zipfile.ZipFile(filename,"r") as zip_ref:
        zip_ref.extractall(target_dir)
#endregion


def isfileExists(filepath):
   return os.path.exists(filepath)
