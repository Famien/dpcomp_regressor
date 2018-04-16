import pickle
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt

models = joblib.load("models_6.pkl")

epsilon_errors = []
errors = []
epsilons = []
predicted_epsilons = []
scales = []
domain_sizes = []
std_devs = []

algs = ["HB", "AHP", "DPCube", "DAWA"]
algs = [algs[1]]
results = {}

dataset_stats = {}
x = None
y = None

data = np.load("/home/famien/Code/pipe/AHP_data_6_test.npy")
print "data 0: ", data[0]

# data_train = data[]
# train = []
# 	test = []
# 	for i in range(len(data)):
# 		if random.random() >= .5:
# 			train.append(i)
# 		else:
# 			test.append(i)

# x = map(lambda x: x[2], data)
# y = map(lambda x: x[6], data)

print len(data)
model = models["AHP"]

for i in range(len(data)):
	stat = data[i][0:6]
	predicted_epsilon = model.predict([stat])
	errors.append(stat[2])
	predicted_epsilons.append(predicted_epsilon)
	epsilon = data[i][6]
	epsilons.append(epsilon)
	scales.append(stat[0])
	domain_sizes.append(stat[1])
	std_devs.append(stat[4])

# for alg in algs:
# 	print alg, " : "
# 	model = models[alg]
# 	data = np.load("/home/famien/Code/pipe/" + alg + "_real_data.npy")
# 	print "len data: ", len(data)
# 	epsilons_for_alg = []
# 	for i in range(len(data)):
# 		stat = data[i][0:6]
# 		# model input/stat fields: scale, domain_size, error, data_range, std_dev, uniform_distance, 
# 		#      			             0         1         2       3           4           5              
# 		epsilon = data[i][6]
# 		predicted_epsilon = model.predict([stat])
# 		percent_error = (abs(predicted_epsilon - epsilon)/epsilon)*100
# 		epsilon_errors.append(percent_error)
# 		epsilons_for_alg.append((epsilon, predicted_epsilon))

# 		epsilons.append(epsilon)
# 		errors.append(stat[2])
# 		predicted_epsilons.append(predicted_epsilon)

# 		stat = tuple(stat)
# 		stat = (stat[0], stat[1], stat[3], stat[4],stat[5]) # don't include 
# 		if stat in dataset_stats.keys():
# 			dataset_stats[stat][alg] = (epsilon, predicted_epsilon[0])
# 		else:
# 			dataset_stats[stat] = {alg : (epsilon, predicted_epsilon[0])}

# 	results[alg] = epsilons_for_alg
# 	print "avg error: ", sum(epsilon_errors)/len(epsilon_errors)

# for dataset_stat in dataset_stats.keys():
# 	alg_epsilon_info = dataset_stats[dataset_stat]
# 	print "epsilons: ", alg_epsilon_info
# 	best_alg = min(alg_epsilon_info, key = lambda x : x[0])
# 	predicted_alg = min(alg_epsilon_info, key = lambda x : x[1])
	#print "best alg: ", best_alg,  " predicted alg: ", predicted_alg

# plt.scatter(errors, epsilons, color='black')
# plt.ylabel('actual epsilon')

plt.scatter(errors, predicted_epsilons, color='black')
plt.ylabel('predicted epsilons')
plt.xlabel('error /100')

# plt.scatter(std_devs, epsilons)
# plt.xlabel('std_devs')
# plt.ylabel('epsilon')

#plt.scatter([x[2] for x in test_X], epsilon_predict, color='blue')
plt.show()


pickle.dump({'results': results, 'dataset_stats' : dataset_stats}, open("real_dataset_stats.p", "wb"))


