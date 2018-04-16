from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pylab
import random
import sys
import math
import pickle
from sklearn.externals import joblib


joblib.dump({}, "models_final.pkl")

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
features = ["scale", "domain_size", "error", "data_range", "std_dev", "uniform_distance"]
models = joblib.load("models.pkl")

print "models: ", models
for alg in algs:
	model = models[alg]

	#print "accuracy: ", regr.score(test_X,test_y)
	print "Alg: ", alg
	print "\tvar explained: ", r2_score(test_y, epsilon_predict)
	print sorted(zip(map(lambda x: float("{0:.2f}".format(round(x, 4))), regr.feature_importances_), features),
	             reverse=True)

	# font = {'family' : 'normal',
	#         'weight' : 'bold',
	#         'size'   : 12}

	# matplotlib.rc('font', **font)

	# plt.figure(figsize=(25,20))

	# plt.bar(plt_x, rf.feature_importances_, width=0.5, color="blue",align='center')
	# plt.gca().set_xticklabels(plt_x, rotation=60 )
	# plt.xticks(plt_x, features)
	# plt.ylabel("relative information")
	# plt.xlabel("features")
	# plt.show()