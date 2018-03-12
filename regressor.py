from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import csv
import random
import sys
import math
import pickle
from sklearn.externals import joblib

#joblib.dump({}, "models.pkl")

models = joblib.load("models.pkl")
print("models: ", models)
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

# train_data_X = map(lambda x: [0, 0, math.log(x[2]), 0, math.log(x[5])], train_data_X)
# train_data_X = map(lambda x: [x[0], x[1], max(0.0000000001, x[2]), x[3],x[4], max(0.0000000001, x[5])], train_data_X)
# train_data_X = map(lambda x: [x[0], x[1], math.log(x[2]), x[3],x[4], x[5]], train_data_X)
# train_data_X = map(lambda x: [ math.log(x[2])], train_data_X)
'''
# fields: scale, domain_size, error, data_range, std_dev, uniform_distance, epsilon
#           0         1         2       3           4           5              6
algs = ["HB", "AHP", "DPCube", "DAWA"]

for alg in algs:

	data = np.load("/home/ubuntu/Code/dpcomp_core/"+alg+"_results_1-5.npy")
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

	for index in train:
		train_X.append(data[index][0:6])
		train_y.append(data[index][6])

	for index in test:
		test_X.append(data[index][0:6])
		test_y.append(data[index][6])

	print("num test data: ", len(train_X))
	#train_data_X = map(lambda x: x[0:6], data)

	X_ = train_X
	y = train_y

	regr = RandomForestRegressor()
	regr.fit(X_,y)

	#models = pickle.loads("models.pickle")
	models = joblib.load("models.pkl")
	models[alg] = regr
	joblib.dump(models, "models.pkl")

	epsilon_predict = regr.predict(test_X)
	print("alg: ", alg)
	#print "accuracy: ", regr.score(test_X,test_y)
	print("var explained: ", r2_score(test_y, epsilon_predict))
	print(regr.feature_importances_)
