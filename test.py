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

patent_data_x = []
patent_data_y = []

with open('training_data_x.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile, delimiter=' ')
	for row in reader:
		new_row = map(lambda x: float(x), row)
		patent_data_x.append(new_row)

with open('training_data_y.csv', 'rb') as csvfile:
	reader = csv.reader(csvfile)
	patent_data_y = map(list, reader)



df = pd.DataFrame(data=patent_data_x)


target = pd.DataFrame(data=patent_data_y)

X = df
y = target


lm = linear_model.LinearRegression()
model = lm.fit(X,y)

predictions = lm.predict(X)

train_data_X = map(lambda x: x[2], patent_data_x)

train_data_X = [ x**2 + random.randrange(-1,1) for x in range(-10,10)]

#print "coef: ", lm.intercept__
m = lm.coef_[0][2]
#b = lm.intercept_
plt.scatter(train_data_X, train_data_Y, color='blue')
#plt.scatter(train_data_X, patent_data_y, color='blue')
#plt.plot([min(train_data_X), max(train_data_X)], [0, m*max(train_data_X) ], 'r')
pylab.show()

#print "predictions, ", predictions
print "variance: ", lm.score(X,y)
print "coef: ", lm.coef_
#print predictions[0:5]

#print data.feature_names
#print len(data.target)
