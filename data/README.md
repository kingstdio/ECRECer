<!--
 * @Author: Zhenkun Shi
 * @Date: 2022-04-26 20:41:28
 * @LastEditors: Zhenkun Shi
 * @LastEditTime: 2022-04-27 09:59:04
 * @FilePath: /DMLF/data/README.md
 * @Description: 
 * 
 * Copyright (c) 2022 by tibd, All Rights Reserved. 
-->

# DATA FOLDER

## Folder Structure

rootfolder/data/datasets/    
rootfolder/data/featureBank/

## Get Data

Bechmark task data set can be downloaded from AWS S3.

### Put in datasets dir
```
wget https://tibd-public-datasets.s3.amazonaws.com/ecrecer/ecrecer_datasets.zip
```


### Put in featureBank dir
```
wget https://tibd-public-datasets.s3.us-east-1.amazonaws.com/ecrecer/data/featureBank/embd_unirep.feather
wget https://tibd-public-datasets.s3.us-east-1.amazonaws.com/ecrecer/data/featureBank/embd_onehot.feather
wget https://tibd-public-datasets.s3.us-east-1.amazonaws.com/ecrecer/data/featureBank/embd_esm33.feather
wget https://tibd-public-datasets.s3.us-east-1.amazonaws.com/ecrecer/data/featureBank/embd_esm32.feather
wget https://tibd-public-datasets.s3.us-east-1.amazonaws.com/ecrecer/data/featureBank/embd_esm0.feather
```
