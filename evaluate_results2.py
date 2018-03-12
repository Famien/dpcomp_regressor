import pickle
from sklearn.externals import joblib
import numpy as np

models = joblib.load("models.pkl")

algs = ["HB", "AHP", "DPCube", "DAWA"]

for alg in algs:
	all_results = []

	epsilon_errors = []
	data = np.load("/home/famien/Code/pipe/"+alg+"mini_test_results_5.npy")
	model = models[alg]

	for i in range(len(data)):
		stat = data[i][0:6]
		# model input fields: scale, domain_size, error, data_range, std_dev, uniform_distance, 
		#      			0         1         2       3           4           5              
		epsilon = data[i][6]
		predicted_epsilon = model.predict([stat])
		percent_error = (abs(predicted_epsilon - epsilon)/epsilon)*100
		epsilon_errors.append(percent_error)

	print "avg error for: ", alg, sum(epsilon_errors)/len(epsilon_errors)

