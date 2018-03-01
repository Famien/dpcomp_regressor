from algorithms import *

def vary_shape(initial_vector, iterations = None):
	data_vectors = []
	if len(initial_vector) == 1:
		return [initial_vector]

	uniform = get_uniform(initial_vector)
	data_vectors.append(uniform)

	avg = uniform[0]
	iterations = min(iterations, (len(uniform)/2) - 1)
	
	left_center = (len(uniform)/2) -1
	ref_vector = uniform
	# slowly move from center to edges
	# slowly move from edges to center
	for i in range(iterations):
		new_vector = ref_vector[:]
		left_move = left_center - i
		right_move = left_center + i + 1

		to_move = (new_vector[left_move])
		new_vector[left_move] -= to_move
		new_vector[right_move] -= to_move
		new_vector[0] += to_move
		new_vector[len(new_vector)-1] += to_move

		data_vectors.append(new_vector)
		ref_vector = new_vector
	
	return data_vectors
