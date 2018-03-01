from algorithms import *
from scipy.stats import kurtosis

def uniform_distance(A):
	total = sum(A)
	avg = total/len(A)
	
	distance = 0
	for element in A:
		distance += abs(avg - element)
		
	scaled_distance = (distance/total)/2
	
	return scaled_distance	

scale = 20787122
domain_size = 2049 
data_vectors = []
# get uniform distribution
# slowly move from center to edges
# slowly move from edges to center

initial_vector = [0]*domain_size
initial_vector[0] = scale

uniform = get_uniform(initial_vector)
data_vectors.append(uniform)

'''
area = (domain_size/2)*uniform[0]
passes = 10

increment = (scale / passes)/2

old_left = uniform[0]

for i in range(passes):
	new_vector = uniform[:]
	center = (len(uniform)/2)
	left = 0
	right = len(uniform) - 1
	x = old_left + increment
	old_left = x
	y = (2*area/(domain_size/2)) - x
	slope = (x-y)/((domain_size/2)-1)
	while left <= center and right >= center:
		new_vector[left] = x
		new_vector[right] = x
		
		left += 1
		right -= 1 	
		x -= slope
	
	data_vectors.append(new_vector)
'''

avg = uniform[0]

iterations = (len(uniform)/2) - 1
left_center = len(uniform)/2 -1 
ref_vector = uniform
for i in range(iterations):
	new_vector = ref_vector[:]
	left_move =left_center - i
	right_move = left_center + i + 1
	
	to_move = (new_vector[left_move])
	new_vector[left_move] -= to_move
	new_vector[right_move] -= to_move
	new_vector[0] += to_move
	new_vector[len(new_vector)-1] += to_move
	
	data_vectors.append(new_vector)
	ref_vector = new_vector
	
	 

for vector in data_vectors:
	print uniform_distance(vector)

