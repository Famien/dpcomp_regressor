import csv
import math
import os
from os import listdir
from os.path import isfile, join
import algorithms as algs
import numpy as np

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
		new_data_vector = algs.merge_data_vector(dataset_vector, new_size)
		data_set_vectors_ext.append(new_data_vector)

'''
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
num_runs = 5V
for histogram_and_domain in histograms:
	histogram = histogram_and_domain[0]
	domain_size = histogram_and_domain[1]
	for epsilon in epsilons:
		for j in range(num_runs):#repeat num_runs times for each epsilon
			private_hist = get_laplace_hist(histogram, epsilon)
			private_histograms_data.append((private_hist, histogram, sum(histogram), domain_size, epsilon))
'''
# 4 get histograms
private_histograms_data= []
epsilons = [0.001, 0.002,0.003, 0.004, 0.005, 0.006, 0.01, 0.02, .1, .2, .3, .5 , 1, 2, 2.5 ,3, 3.5, 4, 5, 7] 
num_runs = 5
num_bins = 50
num_total = len(epsilons)* len(data_set_vectors_ext)
done = 0
for epsilon in epsilons:
	for data_vector in data_set_vectors_ext:
		#num_iterations = 5*math.log(sum(data_vector))
		num_iterations = 2 
		queries = algs.get_queries(num_bins, len(data_vector))
		private_dataset = algs.mwem(data_vector, queries, num_iterations, epsilon)
		histogram, bin_size = algs.get_histogram(data_vector, num_bins)
		private_hist, bin_size = algs.get_histogram(private_dataset, num_bins)
		
		domain_size = len(data_vector)
		private_histograms_data.append((private_hist, histogram, sum(histogram), domain_size, epsilon, data_vector))
		
		done += 1
		if done % 50 == 0:
			print "num done = ", done

# 6. get private histogram error data
error_data = []
for private_histogram_info in private_histograms_data:
	private_histogram = private_histogram_info[0]
	actual_histogram = private_histogram_info[1]
	scale = private_histogram_info[2]
	domain_size = private_histogram_info[3]
	epsilon = private_histogram_info[4]
	error = algs.get_scaled_error(actual_histogram, private_histogram)
	data_vector = private_histogram_data[5]
	error_data.append([scale, domain_size, float(error), np.var(data_vector), epsilon]) 
print "error data: ", error_data
# 7. transform error data


error_data = map(lambda x: [math.log(x[0]), x[1], math.log(x[2]), math.log(x[3])], error_data)

error_data_new = []

#for i in range(len(error_data)):
#	if i % 20 == 0:
#		error_data_new.append(error_data[i])

#error_data = error_data_new

file_name = 'training_data_mwem'
# 8. write error data
with open(file_name + '_x.csv', 'wb') as csvfile:
	with open(file_name + '_y.csv', 'wb') as csvfile2:
		writer = csv.writer(csvfile, delimiter=' ')
		writer2 = csv.writer(csvfile2, delimiter=' ')
		for data_point in error_data:
			writer.writerow(data_point[0: len(data_point)-1])
			epsilon = data_point[len(data_point) -1]
			writer2.writerow([epsilon])
