import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *

# order: [DPcube1D.DPcube1D_engine(),dawa.dawa_engine(),ahp.ahpND_engine(),HB.HB_engine(), privelet.privelet_engine()]

alg = "AHP"
eval_results = pickle.load(open(alg+ "_new_pairs.p", "rb"))
#eval_results = pickle.load(open("/home/famien/Code/pipe/real_dataset_stats.p", "rb"))


desired_errors = []
observed_errors = []

differences = []
for result in eval_results:
	observed_error = result[1]
	desired_error = result[0]	
	differences.append(((desired_error-observed_error)/desired_error)*100)
	#differences.append(desired_error-observed_error)
	desired_errors.append(desired_error*100)
	observed_errors.append(observed_error*100)

print(np.average(differences), " avg actual and predicted: ", np.average(desired_errors), " ", np.average(observed_errors), np.median(observed_errors))

'''
'''
# quartile 


print("total errors: ", len(desired_errors), " ", len(observed_errors))
desired_errors = np.sort(desired_errors)
print("max desired: ", max(desired_errors))
print("max observed_errors " ,max(observed_errors))
upper_q, lower_q = np.percentile(desired_errors, [90, 5])
desired_errors = list(filter(lambda x: x >=lower_q and x <= upper_q, desired_errors))
print("upper q ", upper_q, " lower q: ", lower_q)

observed_errors = np.sort(observed_errors)
upper_q, lower_q = np.percentile(desired_errors, [90, 5])
observed_errors = list(filter(lambda x: x >=lower_q and x <= upper_q, observed_errors))

#desired
# H, X2 = np.histogram( desired_errors, bins = 10, normed = True)
# dx = X2[1] - X2[0]
# F2 = np.cumsum(H)*dx

# #actual
# H3, X3 = np.histogram(observed_errors, bins = 10, normed = True)
# dx = X3[1] - X3[0]
# F3 = np.cumsum(H3)*dx

# plt.subplot(111)

# plt.xlabel('% error', fontsize=16)
# plt.ylabel('Fraction of Data', fontsize=16)
# print("X: ", X2[1:])
# #print(" Y: ", F2)

# X2 = np.insert(X2, 0,1000)
# print("X: ", X2[1:])

# F2 = np.insert(F2,0 ,0)
# p1 = plt.plot(X2[1:], F2, label="Desired Error")
# #print(" Y: ", F2)

# p2 = plt.plot(X3[1:], F3, label="Observed Error")

# Method 2


F1 = np.array(range(len(desired_errors)))/float(len(desired_errors))
plt.plot(desired_errors , F1, label="Desired Error")

F2 = np.array(range(len(observed_errors)))/float(len(observed_errors))
plt.plot(observed_errors, F2, label="Observed Errors")

# #differences

# upper_q, lower_q = np.percentile(differences, [95, 5])
# differences = list(filter(lambda x: x >=lower_q and x <= upper_q, differences))
# differences = np.sort(differences)
# F3 = np.array(range(len(differences)))/float(len(differences))
# plt.plot(differences, F3)

plt.ylim(ymin=0)


#plt.legend(bbox_to_anchor=(.55, 1), loc=4, borderaxespad=2.)

l = legend(loc=7, bbox_transform=gcf().transFigure)

# plt.scatter(errors, predicted_epsilons, color='blue')
# plt.scatter(errors, predicted_epsilons, color='ora')

# plt.xlabel('Relative Difference in Desired Error (%)  and Observed Error (%) ', fontsize=16)
plt.xlabel('% Error', fontsize=16)

plt.ylabel('Fraction of Data', fontsize=16)

plt.title( alg + ' CDF of Errors')

'''



'''


# differences


# differences = np.sort(differences)
# print("len differences: ", len(differences))
# upper_q, lower_q = np.percentile(differences, [95, 10])
# differences = list(filter(lambda x: x >=lower_q and x <= upper_q, differences))

# print("avg differences: ", np.average(differences))

# H4, X4 = np.histogram(differences, bins = 100, normed = True)
# dx = X4[1] - X4[0]
# F4 = np.cumsum(H4)*dx

# plt.xlabel('Difference in Desired Error (%)  and Observed Error (%) ', fontsize=16)
# plt.ylabel('Fraction of Data', fontsize=16)

# plt.plot(X4[1:], F4, label="Difference in Error")
# plt.title('CDF For Difference in Errors')
# plt.legend(bbox_to_anchor=(.55, 1), loc=2, borderaxespad=2.)



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

