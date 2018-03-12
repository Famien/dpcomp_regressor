import csv
import math
import os
from os import listdir
from os.path import isfile, join
import algorithms as algs
import numpy as np
import random
from data_generation import *
'''
1.) get all files

2.) for each file, create list of data

3.) for each dataset create new dataset varying domain size

4.) for each dataset get a histogram

5.) for each histogram get a private histogram

6.) for each private histogram get error data

7.) tranform error data

8.) output error and stats
'''

# 2-3 generate data vectors

def get_synthetic_data(params = None):
	scales = range(1,200000,10000)
	domain_sizes = range(1,300, 100)

	dataset_vectors_ext = []

	done = 0
	for scale in scales:
		for domain_size in domain_sizes:
			dataset_vectors_ext = dataset_vectors_ext + get_datasets_shape(scale, domain_size, iterations = 20)
			done +=1
			if done % 10 == 0:
				print "done: ", done

	return dataset_vectors_ext


# 4 get private histograms
private_histograms_data= []
num_runs = 1
num_bins = 50
reps = 1

done = 0
epsilons1= [float(x)/100 for x in range(1,10)]
epsilons2 = [float(x)/10 for x in range(10, 20)]
epsilons = epsilons1 + epsilons2

num_total = reps* len(dataset_vectors_ext)*len(epsilons)
print "num total: ", num_total

# for data_vector in dataset_vectors_ext: 
# 	print "my scale: ", sum(data_vector)
# 	print "my domain size: ", len(data_vector)

def get_error_data():
	for i in range(reps):
		for data_vector in dataset_vectors_ext:
			for epsilon in epsilons:
			#num_iterations = 5*math.log(sum(data_vector))
				num_iterations = 2 
				queries = algs.get_queries(num_bins, len(data_vector))
				private_dataset = algs.mwem(data_vector, queries, num_iterations, epsilon)
				histogram, bin_size = algs.get_histogram(data_vector, num_bins)
				private_hist, bin_size = algs.get_histogram(private_dataset, num_bins)
				
				# collect statistics
				scale = sum(data_vector)
				domain_size = len(data_vector)
				std_dev = math.sqrt(np.var(data_vector))
				uniform_distance = algs.uniform_distance(data_vector)
				private_histograms_data.append((private_hist, histogram, sum(histogram), domain_size, data_range, std_dev, uniform_distance, epsilon))

				done += 1
				if done % 500 == 0:
					print "num done = ", done

# 6. get private histogram error data
error_data = []
for private_histogram_info in private_histograms_data:
	private_histogram = private_histogram_info[0]
	actual_histogram = private_histogram_info[1]
	scale = private_histogram_info[2]
	domain_size = private_histogram_info[3]
	data_range = private_histogram_info[4]
	std_dev = private_histogram_info[5]
	uniform_distance = private_histogram_info[6]
	epsilon = private_histogram_info[7]
	error = algs.get_scaled_error(actual_histogram, private_histogram)
	error_data.append([scale, domain_size, float(error), data_range, std_dev, uniform_distance, epsilon]) 


# 7. transform error data


#error_data = map(lambda x: [math.log(x[0]), x[1], math.log(x[2]), math.log(x[3])], error_data)
#error_data = map(lambda x: [math.log(x[0]), x[1], math.log(x[2]), math.log(x[3])], error_data)


error_data_new = []

#for i in range(len(error_data)):
#	if i % 20 == 0:
#		error_data_new.append(error_data[i])

#error_data = error_data_new

file_name = 'training_data_mwem'
# 8. write error data

#   0       1          2        3          4              5           6
# scale, domain_size, error, data_range, std_dev, uniform distance, epsilon
with open(file_name + '_x.csv', 'wb') as csvfile:
	with open(file_name + '_y.csv', 'wb') as csvfile2:
		writer = csv.writer(csvfile, delimiter=' ')
		writer2 = csv.writer(csvfile2, delimiter=' ')
		for data_point in error_data:
			writer.writerow(data_point[0: len(data_point)])
			epsilon = data_point[len(data_point) -1]
			writer2.writerow([epsilon])
