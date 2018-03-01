from scipy.stats import kurtosis

a = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]

b = [0,0,20,100,50000,100,20,0,0]

c = [0,0,0,3,2]

d = [2,3,0,0,0]

f = [10000000000,0,0,0,0,0,0,0,0,0,0,0,1000000000]

g = [1,0,0,0,0,0,0,0,0,0,0,0,1]

dists = [a,b,c,d,f, g]

for dist in dists:
	print kurtosis(dist)
