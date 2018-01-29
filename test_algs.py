from  algorithms import *
B = [100,200,300,400,500,600,700]
queries = [(0,3),(4,6)]
num_iterations = 12
epsilon = .001

print "B: ", B
print "A: ", mwem(B, queries, num_iterations, epsilon)



epsilon = .1

print "B: ", B
print "A: ", mwem(B, queries, num_iterations, epsilon)


