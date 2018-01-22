from sklearn import linear_model
from sklearn import datasets
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pylab

'''
algorithms

laplace
'''

train_data_X = []
train_data_y = []

with open('training_data_x.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ')
	for row in reader:
		new_row = map(lambda x: float(x), row)
		train_data_X.append(new_row)

with open('training_data_y.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	train_data_y  = map(list, reader)

df = pd.DataFrame(data=train_data_X)
target = pd.DataFrame(data=train_data_y)

X = df
y = target

lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)

train_data_X = map(lambda x: x[2], train_data_X)

#print "coef: ", lm.intercept__
m = lm.coef_[0][2]
plt.scatter(train_data_X, train_data_y, color='blue')
#plt.scatter(train_data_X, patent_data_y, color='blue')
#plt.plot([min(train_data_X), max(train_data_X)], [0, m*max(train_data_X) ], 'r')
pylab.show()

#print "predictions, ", predictions
print "variance: ", lm.score(X,y)
print "coef: ", lm.coef_
#print predictions[0:5]

#print data.feature_names
#print len(data.target)
