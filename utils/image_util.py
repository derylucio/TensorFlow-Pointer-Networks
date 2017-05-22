# Utils for manipulating images
import image_slicer 	# NOTE: This can only split a max of 99 * 99 pixels
import os.path
import numpy as np
import collections
import matplotlib.pyplot as plt
from scipy.misc import imread, imsave
from PIL import Image

outputExt = ".png"
previewSize = 600   
zeroPad = 2 		# TODO: Change depending on how many pieces slicer can handle
maxWidth = 99

# Returns a 3D pixel representation of the image.
def getImageAsMatrix(filename):
	return imread(filename).astype(np.float32)

# Splits the image into a width * width grid .
def getSplitImageMatrices(filename, width, save=True, destination="."):
	# showImage(filename)
	assert(width <= maxWidth)
	print "Splitting %s into %d-by-%d tiles." % (filename, width, width)
	tiles = image_slicer.slice(filename, width * width, save=False)
	splitImageMatrices = collections.defaultdict()
	if (save):
		basename = os.path.basename(filename)
		prefix = basename.split('.')[0]	# "man.jpg" becomes "man"
		image_slicer.save_tiles(tiles, directory=destination, prefix=prefix)
	split_images = {}
	for n, tile in enumerate(tiles):
		i = n % width
		j = n / width
		splitImageMatrices[(i, j)] = np.array(tile.image.getdata(), \
											np.uint8).reshape(tile.image.size[1], tile.image.size[0], 3)
	return splitImageMatrices

def showImage(filename):
	image_data = imread(filename).astype(np.float32)
	print "Size,  ", image_data.size
	print "Shape, ", image_data.shape
	scaled_image_data = image_data / 255 # imshow requires values between 0 and 1.
	plt.imshow(scaled_image_data)
	plt.show()

# Takes a permutation as a 1-indexed grid and draws the image.
def mergeAndShowImage(prefix, permutation):
	width = len(permutation)
	reducedSize = previewSize / width
	imgWidth, imgHeight = reducedSize, reducedSize
	result = Image.new("RGB", (previewSize, previewSize))
	for i in xrange(len(permutation)):
		for j in xrange(len(permutation[i])):
			x = i * imgWidth
			y = j * imgHeight
			filename = getImageFilename(prefix, permutation[i][j])
			print "Getting filename %s:" % filename
			img = Image.open(filename)
			img.thumbnail((reducedSize, reducedSize), Image.ANTIALIAS)
			imgWidth, imgHeight = img.size
			print('pos {0},{1} size {2},{3}'.format(x, y, imgWidth, imgHeight))
			result.paste(img, (x, y, x + imgWidth, y + imgHeight))
	result.crop((0, 0, (imgWidth + 1) * width, (imgHeight + 1) * width))
	result.show()

def getImageFilename(prefix, (i, j)):
	return prefix + "_" + str(j).rjust(zeroPad, '0') + "_" + \
						  str(i).rjust(zeroPad, '0') + outputExt

def getPixelData(val):
	i, j = val
	print name + "_" + str(j).rjust(zeroPad, '0') + "_" + str(i).rjust(zeroPad, '0') + ".png"
	return image.getImageAsMatrix(getImageFilename(name, val))

# TESTING
# getSplitImageMatrices("rocket.jpg", 4)
# showImage("man.jpeg")
# mergeAndShowImage("man", [[(2, 1), (1, 2)], [(2, 2), (1, 1)]])