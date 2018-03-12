import pickle
from sklearn.externals import joblib
import numpy as np

models = joblib.load("models.pkl")

exp_results = pickle.load( open( "/home/famien/Code/pipe/mini_experiment_results2.p", "rb" ) )

all_results = []

epsilon_errors = []

#print "exp_ results ", exp_results

print len(exp_results)*len(exp_results[0]["runs"])

algs = ["HB", "AHP", "DPCube", "DAWA"]

for alg in
data = np.load("/home/famien/Code/pipe/HBmini_test_results_4.npy")
model = models["HB"]

for i in range(len(data)):
	#dataset = results["dataset"]
	stat = data[i][0:6]
	# stat fields: scale, domain_size,  data_range, std_dev, uniform_distance
	#           0         1           2           4           5          
	'''
	for each run
		whether we predicted the optimally algorithm
		for each alg, how close we were to predicing the error
	'''
	# model input fields: scale, domain_size, error, data_range, std_dev, uniform_distance, 
	#      			0         1         2       3           4           5              
	epsilon = data[i][6]
	#print "actual epsilon : ", epsilonso
	predicted_epsilon = model.predict([stat])
	#print "predicted epsilon: ", predicted_epsilon

	percent_error = (abs(predicted_epsilon - epsilon)/epsilon)*100
	epsilon_errors.append(percent_error)

	#print "percent error: ", percent_error

print "avg error: ", sum(epsilon_errors)/len(epsilon_errors)

