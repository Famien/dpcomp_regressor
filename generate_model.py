from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pylab
import random
import sys
import math
'''
generate model from synthetic data
'''

training_data = sys.argv[1]
with open('training_data_' + training_data + '_y.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	train_data_Y = list(csvreader)
	train_data_Y = map(lambda x: [float(x[0])], train_data_Y)

 
with open('training_data_' + training_data + '_x.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	train_data_X = list(csvreader)
	train_data_X = map(lambda x:  map(lambda y: float(y), x)[0:6], train_data_X)

#data = np.load("/home/famien/Code/dpcomp_core/dpcube_results.npy")
data = train_data_X
# l = map(lambda x: x[0], data)
# print min(l), " ", max(l)

#train_data_X = map(lambda x: x[0:6], data)
train_data_X = map(lambda x: [0, 0, math.log(x[2]), 0, math.log(x[5])], data)
#train_data_Y = map(lambda x: [x[6]], data)
#train_data_X = map(lambda x: [0, 0, x[2], 0, 0, 0], train_data_X)

# print "all: ", len(train_data_X)

# good_indices = []
# for i in range(len(train_data_X)):
# 	if abs(train_data_X[i][2] - .1) <= .02:
# 		good_indices.append(i)

# for i in range(len(train_data_Y)):
# 	if train_data_Y[i][0] == .1:
# 		print train_data_Y[i]
# 		good_indices.append(i)

# newX = []
# newY = []
# for index in good_indices:
# 	newX.append(train_data_X[index])
# 	newY.append(train_data_Y[index])

# # for i in  const_error_X:
# # 	print i
# train_data_X = newX
# train_data_Y = newY

#print "filtered: ", len(const_error_X)
#poly = PolynomialFeatures(degree=2)
#X_ = poly.fit_transform(train_data_X)
X_ = pd.DataFrame(data=train_data_X)
y = pd.DataFrame(data=train_data_Y)
#predict = poly.fit_transform(X)

lm = linear_model.LinearRegression()
model = lm.fit(X_,y)

#predictions = lm.predict(X)

X_2 = map(lambda x: x[2], train_data_X)
#X_1 = map(lambda x: x[2], train_data_X)
print lm.coef_
#print "coef: ", lm.intercept__
m = lm.coef_[0][0]
b = lm.intercept_
plt.scatter(X_2, y, color='blue')
#plt.scatter(train_data_X, patent_data_y, color='blue')
#plt.plot([min(X), max(X)], [b, m*max(X)+b ], 'r')
#plt.plot(X, lm.predict(X_), color='red')
pylab.show()

#print "predictions, ", predictions
print "variance: ", lm.score(X_,y)
print "coef: ", lm.coef_
#print predictions[0:5]

#print data.feature_names
#print len(data.target)
