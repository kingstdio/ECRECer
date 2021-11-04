import numpy
import sys
import nmslib
import time
import math
import pdb

lbl_ft_file = sys.argv[1]
model_file = sys.argv[2]
M = int(sys.argv[3])
efC = int(sys.argv[4])
num_threads = int(sys.argv[5])
num_ft = int(sys.argv[6])
metric_space = sys.argv[7]

start = time.time()
fp = open(lbl_ft_file,'rb')
fp.seek(8)
data = numpy.fromfile(fp,dtype=numpy.float32,count=-1,sep='')
data = numpy.reshape(data,(int(len(data)/num_ft),num_ft))
end = time.time()
start = time.time()
index = nmslib.init(method='hnsw', space=metric_space)
index.addDataPointBatch(data)
index.createIndex({'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC})
end = time.time()
print('Training time of ANNS datastructure = %f'%(end-start))
nmslib.saveIndex(index,model_file)
