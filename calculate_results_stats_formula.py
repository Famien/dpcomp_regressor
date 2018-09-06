import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import math

eval_results = pickle.load(open("/home/famien/Code/pipe/privelet_formula_errors.p", "rb"))
#eval_results = pickle.load(open("/home/famien/Code/pipe/real_dataset_stats.p", "rb"))


starting_errors = []
output_errors = []
formula_starting_errors = []
formula_output_errors = []


differences = []
for result in eval_results:
	starting_error = result[0]
	output_error = result[1]
	formula_starting_epsilon_error = result[2]
	formula_output_epsilon_error = result[3]

	starting_errors.append(starting_error*100)
	output_errors.append(output_error*100)
	differences.append((starting_error-output_error)*100)
	formula_starting_errors.append(math.sqrt((formula_starting_epsilon_error*100))*1.6) # 95th percentile
	formula_output_errors.append(math.sqrt((formula_output_epsilon_error*100))*1.6)


print( " avg actual and predicted: ", np.average(starting_errors), " ", np.average(formula_starting_errors), np.average(formula_output_errors))

'''
'''
# quartile 


starting_errors = np.sort(starting_errors)
output_errors = np.sort(output_errors)
formula_starting_errors = np.sort(formula_starting_errors)
formula_output_errors = np.sort(formula_output_errors)
print("max desired: ", max(starting_errors))
print("max observed_errors " ,max(formula_starting_errors))
print("max observed_errors " ,max(formula_output_errors))

'''
quartiles
# print("upper q ", upper_q, " lower q: ", lower_q)


# observed_errors = np.sort(observed_errors)


'''
upper_q, lower_q = np.percentile(starting_errors, [95, 5])
starting_errors = list(filter(lambda x: x >=lower_q and x <= upper_q, starting_errors))

upper_q, lower_q = np.percentile(output_errors, [95, 5])
output_errors = list(filter(lambda x: x >=lower_q and x <= upper_q, output_errors))

upper_q, lower_q = np.percentile(formula_starting_errors, [95, 5])
formula_starting_errors = list(filter(lambda x: x >=lower_q and x <= upper_q, formula_starting_errors))

upper_q, lower_q = np.percentile(formula_output_errors, [95, 5])
formula_output_errors = list(filter(lambda x: x >=lower_q and x <= upper_q, formula_output_errors))


F1 = np.array(range(len(starting_errors)))/float(len(starting_errors))
plt.plot(starting_errors , F1, label="Starting Error")

F2 = np.array(range(len(formula_starting_errors)))/float(len(formula_starting_errors))
plt.plot(formula_starting_errors, F2, label="Formula Predicted Errors")

plt.ylim(ymin=0)

plt.legend(loc=7, bbox_transform=gcf().transFigure)


'''
#starting
# H, X2 = np.histogram(starting_errors, bins = 100, normed = True)
# dx = X2[1] - X2[0]
# F2 = np.cumsum(H)*dx

# # output
# H3, X3 = np.histogram(output_errors, bins = 100, normed = True)
# dx = X3[1] - X3[0]
# F3 = np.cumsum(H3)*dx

# # formula on output
# H4, X4 = np.histogram(formula_output_errors, bins = 100, normed = True)
# dx = X4[1] - X4[0]
# F4 = np.cumsum(H4)*dx

# # formula on actual
# H5, X5 = np.histogram(formula_starting_errors, bins = 100, normed = True)
# dx = X5[1] - X5[0]
# F5 = np.cumsum(H5)*dx
'''



#plt.subplot(111)

# plt.xlabel('% error', fontsize=16)
# plt.ylabel('Fraction of Data', fontsize=16)

# # p1 = plt.plot(X2[1:], F2, label="Starting Error")
# # p2 = plt.plot(X3[1:], F3, label="Output Error")
# # p3 = plt.plot(X4[1:], F4, label="Forumla Predicted Error (Original Epsilon 95th Percentile)")
# # p4 = plt.plot(X4[1:], F5, label="Forumla Output Error (Output Epsilon 95th Percentile)")

# plt.legend(bbox_to_anchor=(.55, 1), loc=2, borderaxespad=2.)

# l = legend(bbox_to_anchor=(0, 0, 1, 1), bbox_transform=gcf().transFigure)

# # plt.scatter(errors, predicted_epsilons, color='blue')
# # plt.scatter(errors, predicted_epsilons, color='ora')


# plt.title('CDF For Errors After Running Private Algs')

'''



'''


# differences


differences = np.sort(differences)
print("len differences: ", len(differences))
upper_q, lower_q = np.percentile(differences, [95, 5])
differences = list(filter(lambda x: x >=lower_q and x <= upper_q, differences))

print("avg differences: ", np.average(differences))

H4, X4 = np.histogram(differences, bins = 100, normed = True)
dx = X4[1] - X4[0]
F4 = np.cumsum(H4)*dx

plt.xlabel('% Error ', fontsize=16)
plt.ylabel('Fraction of Data', fontsize=16)

#plt.plot(X4[1:], F4, label="Difference in Error")
plt.title('CDF Privelet Errors')

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

