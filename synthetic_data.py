import numpy
import csv
import math
import os
from os import listdir
from os.path import isfile, join
import algorithms as algs
import random
from data_generation import *

'''
mypath = join(os.getcwd(), "datafiles/1D")
mypath = "/home/famien/Code/pipe/1D"
'''
data_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

# 2. get data vectors from files
dataset_vectors = []

for data_file in data_files:
	data_from_file = np.load(data_file)
	dataset_vectors.append(data_from_file)

print "len dataset: ", len(dataset_vectors)
# 3. extend vectors by merging domain buckets
dataset_vectors_ext = []
#new_sizes = [float(x)/100 for x in range(1,100,12)]
new_sizes = [.1, .25]
for dataset_vector in dataset_vectors:
	dataset_vectors_ext.append(dataset_vector)
	for new_size in new_sizes:
		new_num_bins = new_size * len(dataset_vector)
		if new_num_bins <= 50:
			continue # no need for execessively small datasets?
		new_data_vector = algs.merge_data_vector(dataset_vector, new_size)
		dataset_vectors_ext.append(new_data_vector)
print "len dataset: ", len(dataset_vectors_ext)

done = 0
for dataset in dataset_vectors_ext:
		new_vector = vary_shape(dataset, iterations = 4)
		dataset_vectors_ext = dataset_vectors_ext + new_vector
		done +=1
		if done % 1000 == 0:
			print("done: ", done)

print "len dataset: ", len(dataset_vectors_ext)
			
#vary scale
for dataset in dataset_vectors_ext:
		new_vectors = vary_scale(dataset, [.25,4])
		dataset_vectors_ext = dataset_vectors_ext + new_vectors
		done +=1
		if done % 1000 == 0:
			print "done: ", done

print "len dataset: ", len(dataset_vectors_ext)

print "num non synthetic: ", len(dataset_vectors_ext)
# pure synthetic data
# scales = range(1,200000,11000)
# domain_sizes = range(1,3000, 900)

# print "num to make: ", len(scales)*len(domain_sizes)
# done = 0
# for scale in scales:
# 	for domain_size in domain_sizes:
# 		dataset_vector = [0]*domain_size
# 		dataset_vector[0] = scale
# 		dataset_vectors_ext = dataset_vectors_ext + vary_shape(dataset, iterations = 5)
# 		done +=1
# 		if done % 1000 == 0:
# 			print "done: ", done
# print "total synthetic datasets: ", len(dataset_vectors_ext)

data = numpy.array(dataset_vectors_ext)

numpy.save("DATA6", data)

#DATA5 most recent
