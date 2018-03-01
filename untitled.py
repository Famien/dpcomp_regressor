import csv
import math
import os
from os import listdir
from os.path import isfile, join
import algorithms as algs
import numpy as np
import random
from data_generation import *

mypath = join(os.getcwd(), "1d")
data_files = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]


# 2. get data vectors from files

dataset_vectors = []

for data_file in data_files:
	data_from_file = np.loadtxt(data_file, delimiter=' ')
	dataset_vectors.append(data_from_file)


# 3. extend vectors by merging domain buckets

data_set_vectors_ext = []
new_sizes = [.5, .25, .125, .1, .05, .04, .01, .001, .002, .003]
for dataset_vector in dataset_vectors:
	for new_size in new_sizes:
		new_data_vector = algs.merge_data_vector(dataset_vector, new_size)
		data_set_vectors_ext.append(new_data_vector)
		

scales = range(1,200000,10000)
domain_sizes = range(1,300, 100)

dataset_vectors_ext = []

print "num to make: ", len(scales)*len(domain_sizes)
done = 0
for scale in scales:
	for domain_size in domain_sizes:
		dataset_vectors_ext = dataset_vectors_ext + get_datasets_shape(scale, domain_size, iterations = 20)
		done +=1
		if done % 10 == 0:
			print "done: ", done