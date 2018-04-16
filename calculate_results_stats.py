import pickle
import numpy as np
import matplotlib.pyplot as plt

eval_results = pickle.load(open("/home/famien/Code/pipe/error_pairs_real_data.p", "rb"))
#eval_results = pickle.load(open("/home/famien/Code/pipe/real_dataset_stats.p", "rb"))


desired_errors = []
observed_errors = []

differences = []
for result in eval_results:
	observed_error = result[1]
	desired_error = result[0]
	differences.append(abs(result[0] - result[1]))

	desired_errors.append(desired_error)
	observed_errors.append(observed_error)

print  np.average(differences), " avg actual and predicted: ", np.average(desired_errors), " ", np.average(observed_errors), np.median(observed_errors)

X1 = np.sort(desired_errors)
F1 = np.array(range(len(X1)))/float(len(X1))

#desired
H, X2 = np.histogram( desired_errors, bins = 10, normed = True)
dx = X2[1] - X2[0]
F2 = np.cumsum(H)*dx

#actual
H3, X3 = np.histogram(observed_errors, bins = 10, normed = True)
dx = X3[1] - X3[0]
F3 = np.cumsum(H3)*dx

#plt.plot(X1, F1)

ax = plt.subplot(111)
ax.plot(X2[1:], F2, label="Desired Error")
ax.plot(X3[1:], F3, label="Observed Error")

plt.title('CDF For Errors After Running Private Algs')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),  shadow=True, ncol=2)

#plt.hist(desired_errors, bins = 1000)
#plt.hist(observed_errors, bins = 10)

plt.show()




# algs = ["HB", "AHP", "DPCube", "DAWA"]

# for alg in algs:
# 	alg_results = eval_results[alg]
# 	differences = []
# 	epsilons = []
# 	predicted_epsilons = []
# 	for result in alg_results:
# 		actual = result[0]
# 		predicted = result[1][0]
# 		epsilons.append(actual)
# 		predicted_epsilons.append(predicted)
# 		differences.append(abs(actual-predicted))

# 	print "alg: ", alg, " avg difference: ", numpy.average(differences), " avg actual and predicted: ", numpy.average(epsilons), " ", numpy.average(predicted_epsilons)

# for dataset_stat in  eval_results['dataset_stat'].keys():

