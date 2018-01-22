import numpy as np
import csv
import math
import os
from os import listdir
from os.path import isfile, join

def get_histogram(counts, num_bins):
	'''
		creates a histogram of array counts with num_bins bins
		returns histogram and bin_size
	'''
	hist = []
	domain = len(counts)
	bin_size = max(1, domain/ num_bins) # make sure we have at least 1 bin
	bin_sum = 0
	for i in range(len(counts)):
		bin_sum += counts[i]
		if i % bin_size == 0:
			hist.append(bin_sum)
			bin_sum =0
	hist.append(bin_sum)	
	'''	
	for i in range(num_bins):
		sum_i = i*bin_size
		bucket_sum = 0
		while sum_i < (i*bin_size + bin_size) and sum_i < len(counts):
			bucket_sum += counts[sum_i]
			sum_i += 1	
		hist.append(bucket_sum)
	'''
	
	return hist, bin_size

def get_scaled_error(actual_counts, expected_counts):
	total_error = 0
	for i in range(len(actual_counts)):
		total_error += abs(actual_counts[i] - expected_counts[i])
	return total_error / sum(actual_counts)

def get_laplace_noise(e):
        loc, scale = 0., 1.0/float(e)
        return np.random.laplace(loc, scale, 1)

def get_laplace_hist(counts, e):
	'''
		takes a histogram, counts, and epsilon e
		returns private histogram
	'''
	output = []
	for i in range(len(counts)):
		new_count = counts[i] + get_laplace_noise(e)[0]
		output.append(new_count)
	return output

def merge_data_vector(data_vector, new_size):
	new_num_bins = new_size * len(data_vector)
	merge_size = len(data_vector)/new_num_bins
	new_data_vector = []
	
	bin_sum = 0
	for i in range(len(data_vector)):
		bin_sum += data_vector[i]
		if i % merge_size == 0:
			new_data_vector.append(bin_sum)
			bin_sum = 0
	new_data_vector.append(bin_sum)
	return new_data_vector

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

# 1. get files

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
		new_data_vector = merge_data_vector(dataset_vector, new_size)
		data_set_vectors_ext.append(new_data_vector)

# 4. get histograms
histograms = []
num_bins = 50
for data_vector in data_set_vectors_ext:
	domain_size = len(data_vector)
	histogram, bin_size = get_histogram(data_vector, num_bins)
	histograms.append((histogram, domain_size))

# 5. get private histograms
private_histograms_data = []
epsilons = [0.001, 0.002,0.003, 0.004, 0.005, 0.006, 0.01, 0.02, .1, .2, .3, .5 , 1, 2, 2.5 ,3, 3.5, 4, 5,7] 
num_runs = 5
for histogram_and_domain in histograms:
	histogram = histogram_and_domain[0]
	domain_size = histogram_and_domain[1]
	for epsilon in epsilons:
		for j in range(num_runs):#repeat num_runs times for each epsilon
			private_hist = get_laplace_hist(histogram, epsilon)
			private_histograms_data.append((private_hist, histogram, sum(histogram), domain_size, epsilon))

# 6. get private histogram error data
error_data = []
for private_histogram_info in private_histograms_data:
	private_histogram = private_histogram_info[0]
	actual_histogram = private_histogram_info[1]
	scale = private_histogram_info[2]
	domain_size = private_histogram_info[3]
	epsilon = private_histogram_info[4]
	error = get_scaled_error(actual_histogram, private_histogram)
	error_data.append([scale, domain_size, float(error), epsilon]) 

# 7. transform error data

error_data = map(lambda x: [math.log(x[0]), x[1], math.log(x[2]), math.log(x[3])], error_data)

# 8. write error data
with open('training_data_x.csv', 'wb') as csvfile:
	with open('training_data_y.csv', 'wb') as csvfile2:
		writer = csv.writer(csvfile, delimiter=' ')
		writer2 = csv.writer(csvfile2, delimiter=' ')
		for data_point in error_data:
			writer.writerow(data_point[0: len(data_point)-1])
			epsilon = data_point[len(data_point) -1]
			writer2.writerow([epsilon])
