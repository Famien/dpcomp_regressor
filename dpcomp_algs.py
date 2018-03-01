import sys
sys.path.insert(0, '/home/famien/Code/dpcomp_core/')
import dpcomp_core
import numpy as np

# get data
x = np.array(np.load("DATA1.npy")[300])

# Instantiate algorithm
a = DPcube1D.DPcube1D_engine()

# Instantiate workload
x = workload.Prefix1D(domain_shape_int=domain1)

# Calculate noisy estimate for x
x_hat = a.Run(w, x, epsilon, seed)

print("x: ", x.shape)
print("x hat: ", x_hat.shape)

num_bins = 50

histogram, bin_size = algs.get_histogram(x, num_bins)
private_hist, bin_size = algs.get_histogram(x_hat, num_bins)
			

error = algs.get_scaled_error(histogram, private_hist)
print(error)