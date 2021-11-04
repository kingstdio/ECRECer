import numpy
import sys
import nmslib
import time
import math
from multiprocessing import Process

def write_knn_out(out_dir,write_dist,num_inst,nbrs,batch_no,metric_space):
	with open('%s/%d'%(out_dir,batch_no),'w') as fp:
		fp.write('%d %d\n'%(len(nbrs),num_inst))
		if write_dist == 1:
			for j in range(0,len(nbrs)):
				temp = {}
				flag = 0
				for k in range(0,len(nbrs[j][0])):
					if metric_space == 'l2':
						temp[nbrs[j][0][k]] = nbrs[j][1][k]
					else:
						temp[nbrs[j][0][k]] = 1-nbrs[j][1][k]
				for t in sorted(temp):
					if flag ==0:
						fp.write('%d:%f'%(t,temp[t]))
						flag = 1
					else:
						fp.write(' %d:%f'%(t,temp[t]))
				fp.write('\n')
		else:
			for j in range(0,len(nbrs)):
				temp = {}
				flag = 0
				for k in range(0,len(nbrs[j][0])):
					temp[nbrs[j][0][k]] = 1
				for t in sorted(temp):
					if flag ==0:
						fp.write('%d'%(t))
						flag = 1
					else:
						fp.write(' %d'%(t))
				fp.write('\n')

tst_ft_file = sys.argv[1]
model_file = sys.argv[2]
num_ft = int(sys.argv[3])
num_lbls = int(sys.argv[4])
efS = int(sys.argv[5])
num_nbrs = int(sys.argv[6])
write_dist = int(sys.argv[7])
out_dir = sys.argv[8]
num_thread = int(sys.argv[9])
num_out_threads = int(sys.argv[10])
metric_space = sys.argv[11]

index = nmslib.init(method='hnsw', space=metric_space)
nmslib.loadIndex(index,model_file)

index.setQueryTimeParams({'efSearch': efS, 'algoType': 'old'})

start = time.time()
fp = open(tst_ft_file,'rb')
fp.seek(8)
query = numpy.fromfile(fp,dtype=numpy.float32,count=-1,sep='')
query = numpy.reshape(query,(int(len(query)/num_ft),num_ft))
fp.close()
end = time.time()
start = time.time()
nbrs = index.knnQueryBatch(query, k=num_nbrs, num_threads = num_thread)
end = time.time()
print('Time taken to find approx nearest neighbors = %f'%(end-start))

batch_size = int(math.ceil(float(len(nbrs))/float(num_out_threads)))
for i in range(num_out_threads):
	Process(target=write_knn_out, args=(out_dir,write_dist,num_lbls,nbrs[i*batch_size:min((i+1)*batch_size,len(nbrs))],i,metric_space)).start()
