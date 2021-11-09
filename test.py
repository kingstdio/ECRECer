import numpy as np
import pandas as pd
import time
import datetime
import sys
import os
from tqdm import tqdm
from functools import reduce

import tools.funclib as funclib

sys.path.append("../")
import benchmark_train as btrain
import benchmark_test as btest
import config as cfg
import benchmark_evaluation as eva


from pandarallel import pandarallel #  import pandaralle
pandarallel.initialize() # init


if __name__ =='__main__':
    print('0k')
    test = pd.read_feather(cfg.DATADIR+'task3/test.feather')
    test.iloc[np.r_[105:107],:].parallel_apply(lambda x: funclib.split_ecdf_to_single_lines_pr_record(x)  , axis=1)
    print(len(test))


