# Utils for Evaulating the Fitness function.
import numpy as np
import collections
import image_util # Sanya wrote this
import time

zeroPad = 2
filename = "man.jpeg"
name = filename.split(".")[0]

# Returns the fitness score between two cells. 
def getFitness(chromosome):
	rval, dval = 0.0, 0.0
	width = len(chromosome)
	for row in xrange(len(chromosome)):
		for col in xrange(len(chromosome[row]) - 1):
			left = row * width + col
			right = left + 1
			rval += disimilarity[(left, right, "r")]
	for row in xrange(len(chromosome) - 1):
		for col in xrange(len(chromosome[row])):
			top = row * width + col
			bottom = top + width
			dval += disimilarity[(top, bottom, "d")]
	return rval + dval

def getRightDisimilarity(left, right):
	result = 0.0
	width, height, depth = left.shape
	for k in xrange(height):
		for b in xrange(depth): 
			result += np.square(int(left[width - 1][k][b]) - int(right[0][k][b]))
	return np.sqrt(result)

def getDownDisimilarity(top, bottom):
	result = 0.0
	width, height, depth = top.shape
	for k in xrange(width):
		for b in xrange(depth): 
			result += np.square(int(top[k][height - 1][b]) - int(bottom[k][0][b]))
	return np.sqrt(result)

# TODO: Dank
def getPixelData(val):
	i, j = val
	filename = getImageFilename(name, val)
	print "Getting Pixel Data for %s" % filename
	return image_util.getImageAsMatrix(filename)

def cacheDisimilarityScores(filename, width):
	print "Calculating disimilarity for %s into %d-by-%d images." % (filename, width, width)
	matrices = image_util.getSplitImageMatrices(filename, width, ".")
	n = width * width
	disimilarityScores = collections.defaultdict()
	print "Storing Right-Down Similarities."
	for x in xrange(n):
		for y in xrange(n):
			rowx = x / width
			colx = x % width
			rowy = y / width
			coly = y % width
			disimilarityScores[(x, y, "r")] = getRightDisimilarity(matrices[(rowx, colx)], matrices[(rowy, coly)])
			disimilarityScores[(x, y, "d")] = getDownDisimilarity(matrices[(rowx, colx)], matrices[(rowy, coly)])
			# print "(%d, %d), (%d, %d), r: %d" % (rowx, colx, rowy, coly, disimilarityScores[(x, y, "r")])
			# print "(%d, %d), (%d, %d), d: %d" % (rowx, colx, rowy, coly, disimilarityScores[(x, y, "d")])
	print "Done storing Right-Down Similarities for %s!" % filename
	return disimilarityScores

disimilarity = cacheDisimilarityScores(filename, 16)
# print "Fitness: ", getFitness([[(1,1), (1,2)],[(2,1), (2,2)]])

