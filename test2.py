from sklearn import linear_model
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import pylab
import random
'''
algorithms

laplace
'''


train_data_Y = [ x**2 + random.randrange(-2,2) for x in range(-10,10)]
X = range(-10, 10)
train_data_X = map(lambda x: [x, x**2], range(-10, 10))

poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(train_data_X)
y = pd.DataFrame(data=train_data_Y)
#predict = poly.fit_transform(X)

lm = linear_model.LinearRegression()
model = lm.fit(X_,y)

#predictions = lm.predict(X)


print lm.coef_
#print "coef: ", lm.intercept__
m = lm.coef_[0][0]
b = lm.intercept_
plt.scatter(X, train_data_Y, color='blue')
#plt.scatter(train_data_X, patent_data_y, color='blue')
#plt.plot([min(X), max(X)], [b, m*max(X)+b ], 'r')
plt.plot(X, lm.predict(X_), color='red')
pylab.show()

#print "predictions, ", predictions
print "variance: ", lm.score(X_,y)
print "coef: ", lm.coef_
#print predictions[0:5]

#print data.feature_names
#print len(data.target)
