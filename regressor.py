from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import csv
import random
import sys
import math
import pickle
from sklearn.externals import joblib

'''
generate model from synthetic data
'''

'''
csv data

training_data = sys.argv[1]
with open('training_data_' + training_data + '_y.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	train_data_Y = list(csvreader)
	train_data_Y = map(lambda x: float(x[0]), train_data_Y)

 
with open('training_data_' + training_data + '_x.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	train_data_X = list(csvreader)
	train_data_X = map(lambda x:  map(lambda y: float(y), x)[0:6], train_data_X)

# train_data_X = ap(lambda x: [0, 0, math.log(x[2]), 0, math.log(x[5])], train_data_X)
# train_data_X = map(lambda x: [x[0], x[1], max(0.0000000001, x[2]), x[3],x[4], max(0.0000000001, x[5])], train_data_X)
# train_data_X = map(lambda x: [x[0], x[1], math.log(x[2]), x[3],x[4], x[5]], train_data_X)
# train_data_X = map(lambda x: [ math.log(x[2])], train_data_X)
'''
# fields: scale, domain_size, error, data_range, std_dev, uniform_distance, epsilon
#           0         1         2       3           4           5              6
algs = ["HB", "AHP", "DPCube", "DAWA"]
algs = ["Privelet"]
features = ["scale", "domain_size", "error", "data_range", "std_dev", "uniform_distance"]

for alg in algs:
	data = np.load("/home/famien/Code/pipe/"+alg+"_data_6_new.npy")
	'''
	data = np.load("/home/famien/Code/pipe/"+alg+"_data_6.npy")
	data = np.load("/home/ubuntu/Code/dpcomp_core/"+alg+"_results_1-5.npy")
	'''
	'''
	split into train and test data

	'''
	train = []
	test = []
	for i in range(len(data)):
		if random.random() >= .5:
			train.append(i)
		else:
			test.append(i)

	train_X = []
	train_y = []
	test_X = []
	test_y = []

	all_train_data = []
	all_test_data = []

	print("total len: ", len(data))

	for index in train:
		train_X.append(data[index][0:6])
		train_y.append(data[index][6])
		all_train_data.append(data[index])

	for index in test:
		test_X.append(data[index][0:6])
		test_y.append(data[index][6])
		all_test_data.append(data[index])

	np.save( "/home/famien/Code/pipe/"+alg+"_data_6_train.npy", all_train_data)
	np.save("/home/famien/Code/pipe/"+alg+"_data_6_test.npy", all_test_data)

	# train_X = map(lambda x: [x[0], x[1], max(0.0000000001, x[2]), x[3],x[4], max(0.0000000001, x[5])], train_X)
	# test_X = map(lambda x: [x[0], x[1], max(0.0000000001, x[2]), x[3],x[4], max(0.0000000001, x[5])], test_X)

	print("train len: ", len(train_X))

	X_ = train_X
	y = train_y

	regr = RandomForestRegressor(oob_score = True, max_depth=12)
	#regr = DecisionTreeRegressor(random_state=0)
	regr.fit(X_,y)

	#models = pickle.loads("models.pickle")
	# try:
	# 	models = joblib.load("models_6.pkl")
	# except IOError:
	# 	models = {}
	
	models = {}
	models[alg] = regr
	joblib.dump(models, "models_6.pkl")

	#print "accuracy: ", regr.score(test_X,test_y)
	print("Alg: ", alg)
	#print "\tvar explained: ", r2_score(test_y, epsilon_predict)
	print("out of bag: ", regr.oob_score_)
	print(sorted(zip(map(lambda x: float("{0:.2f}".format(round(x, 4))), regr.feature_importances_), features),
	             reverse=True))

	epsilon_predict_test = regr.predict(test_X)
	epsilon_predict_train = regr.predict(train_X)

	print ("mean squared error train: ", mean_squared_error(train_y, epsilon_predict_train))
	print ("train average, median", sum(epsilon_predict_train)/len(epsilon_predict_train), sorted(epsilon_predict_train)[int(len(epsilon_predict_train)/2)])
	print ("mean squared error test: ", mean_squared_error(test_y, epsilon_predict_test))
	print ("train average, median", sum(epsilon_predict_test)/len(epsilon_predict_test), sorted(epsilon_predict_test)[int(len(epsilon_predict_test)/2)])
