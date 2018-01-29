from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pylab
import random
import sys
'''
algorithms

laplace
'''

'''
test
'''
train_data_Y = [ x**2 + random.randrange(-2,2) for x in range(-10,10)]
X = range(-10, 10)
train_data_X = map(lambda x: [x, x**2], range(-10, 10))

'''
generate model from synthetic data
'''

training_data = sys.argv[1]
print "training _data: ", training_data
with open('training_data_' + training_data + '_y.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	train_data_Y = list(csvreader)

 
with open('training_data_' + training_data + '_x.csv', 'rb') as csvfile:
	csvreader = csv.reader(csvfile, delimiter=' ')
	train_data_X = list(csvreader)

#poly = PolynomialFeatures(degree=2)
#X_ = poly.fit_transform(train_data_X)
X_ = pd.DataFrame(data=train_data_X)
y = pd.DataFrame(data=train_data_Y)
#predict = poly.fit_transform(X)

lm = linear_model.LinearRegression()
model = lm.fit(X_,y)

#predictions = lm.predict(X)

X_2 = map(lambda x: x[2], train_data_X)
print lm.coef_
#print "coef: ", lm.intercept__
m = lm.coef_[0][0]
b = lm.intercept_
plt.scatter(X_2, train_data_Y, color='blue')
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
