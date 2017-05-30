import numpy as np
from fitness_vectorized import fitnessScore

def averageFitness(pieces, dims):
	_, numCols = dims
	totalScore = 0.0
	num_neighbors = 0
	for index, curr_piece in enumerate(pieces):
		next_ind = index + 1
		if(next_ind % numCols != 0):
			right_piece = pieces[next_ind]
			#print curr_piece, right_piece, "R"
			totalScore += fitnessScore(curr_piece, right_piece, "r")
			num_neighbors += 1
		next_ind = index + numCols
		if(next_ind < len(pieces)):
			down_piece = pieces[next_ind]
			totalScore += fitnessScore(curr_piece, down_piece, "d")
			#print curr_piece, down_piece, "D"
			num_neighbors += 1
	return totalScore / num_neighbors

def NeighborAccuracy(originalPerm, proposedPerm, dims):
	_, numCols = dims
	num_correct = 0.0
	total_neighbors = 0.0
	for index, piece_num in enumerate(originalPerm):
		next_ind = index + 1
		prop_index = np.where(proposedPerm == piece_num)[0]
		if(next_ind % numCols != 0):
			if len(prop_index) > 1 :
				prop_right =  prop_index[0] + 1
			else:
				prop_right = prop_index + 1 
			num_correct += 1 if (prop_right % numCols != 0) and  proposedPerm[prop_right] == originalPerm[next_ind]  else 0
			total_neighbors += 1

		next_ind = index + numCols
		if(next_ind < len(originalPerm)):
			if len(prop_index) > 1 :
				prop_down = prop_index[0] + numCols 
			else:
				prop_down = prop_index + numCols
			num_correct += 1 if (prop_down < len(originalPerm)) and proposedPerm[prop_down] == originalPerm[next_ind] else 0
			total_neighbors += 1
	return num_correct/total_neighbors


def directAccuracy(originalPerm, proposedPerm):
	diff = originalPerm - proposedPerm
	return sum(diff == 0)*1.0 / len(diff)


# # TEST NEIGHBOR ACC
#originalPerm = np.array(range(9))
#proposedPerm = np.array([0, 1, 4, 3, 2, 5, 6, 7, 8])
#dims = (3, 3)
#assert NeighborAccuracy(originalPerm, proposedPerm, dims) == 0.5

#originalPerm = np.array(range(9))
#proposedPerm = np.array([8, 6, 7, 3, 4, 5, 1, 2, 0])
#dims = (3, 3)
#print(NeighborAccuracy(originalPerm, proposedPerm, dims))  # should be 0.33333.
