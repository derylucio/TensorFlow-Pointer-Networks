import numpy as np
import scipy.misc
import scipy.ndimage
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize
import multiprocessing
import glob


sys.path.append('utils/')
import fitness_vectorized as fv

NUM_TEST = 8
NUM_TRAIN = 16
NUM_VAL = 8
NUM_DATA = NUM_TEST + NUM_TRAIN + NUM_VAL
DIMS=(64, 64, 3)

numRows, numCols = (3, 3)

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

POOL_SIZE = 8		
NUM_CATS = 2

assert(POOL_SIZE >= min(NUM_TRAIN, NUM_TEST, NUM_VAL))		# Crashes otherwise. 

# For now, assert that this is evenly divisible. 
assert NUM_TEST % NUM_CATS == 0, "{0:d} does not evenly divide {1:d}".format(NUM_CATS, NUM_TRAIN)
assert NUM_VAL % NUM_CATS == 0, "{0:d} does not evenly divide {1:d}".format(NUM_CATS, NUM_VAL)
assert NUM_TRAIN % NUM_CATS == 0, "{0:d} does not evenly divide {1:d}".format(NUM_CATS, NUM_TRAIN)


def getData(puzzle_height, puzzle_width, use_cnn=False):
	'''
	returns 
	data:   data['val'] = (x_val, onehot_cat, y_val, val_seq_len)
			data['train'] = (x_train, onehot_cat, y_train, train_seq_len)
			data['test'] = (x_test, onehot_cat, y_test, test_seq_len) 
			onehot_cat = One Hot vector of categories or *None* if we're not training on category data
			x_train = [batch1, batch2, ... batchn] - each batch should be the image
			seq_len = [batch_size, ] # for each image in that batch, the number of pieces it is cut int
	'''
	H, W = puzzle_height, puzzle_width
	train = loadImages(TRAIN_DIR, NUM_TRAIN, H, W, dims=DIMS)
	val = loadImages(VAL_DIR, NUM_VAL, H, W, dims=DIMS)
	test = loadImages(TEST_DIR, NUM_TEST, H, W, dims=DIMS)
	
	train = shuffleAndReshapeData(train, keep_shape=use_cnn)
	val = shuffleAndReshapeData(val, keep_shape=use_cnn)
	test = shuffleAndReshapeData(test, keep_shape=use_cnn)

	return {
      'train': train,  
      'val'  : val, 
      'test' : test
    }

def getReshapedImages(args):
	imgList, H, W, dims = args
	new_list = []
	for i, img in enumerate(imgList):
		large_width, large_height, large_depth = H * dims[0], W * dims[1], dims[2]
		resized_img = np.array(resize(img, (large_width, large_height, large_depth), preserve_range=True, mode='reflect'))#.astype(dtype=np.uint8)
		new_list.append(resized_img)
	return new_list

def readImg(filename):
	return scipy.ndimage.imread(filename)

def shuffleAndReshapeData(args, keep_shape=True):
	'''
	Splits and preprocessed dimension-formatted data into 
	train, test and validation data. 
	'''
	X, cats, fnames = args
	N, L, W, H, C = X.shape 		# TODO: Check the updated shape. 
	np.random.seed(231)

	X_shuff = np.empty_like(X)
	y_shuff = np.zeros((N, L), dtype=np.uint8)
	y_shuff += np.arange(L, dtype=np.uint8)

	print("Shuffling Data.")
	idx_shuff = np.array(np.random.permutation(X.shape[0]))
	print(idx_shuff)
	X_shuff = X[idx_shuff]
	cats_shuff = cats[idx_shuff]
	fnames_shuff = fnames[idx_shuff]
	assert(np.sum(X - X_shuff) != 0)

	for i in np.arange(X_shuff.shape[0]):
		np.random.shuffle(y_shuff[i])
		X_shuff[i,:] = X_shuff[i,y_shuff[i]]
	print("Shuffled Data.")

	if not keep_shape:
		X = X.reshape(N, L, -1)		# TODO: Update once verified. 
		print("Reshaped to new shape {0}.".format(X.shape))
	
	y_onehot_shuff = np.where(y_shuff[:,:,np.newaxis] == np.arange(L), 1, 0) 
	seq = np.ones((len(X_shuff))) * L
	
	return X_shuff, y_onehot_shuff, cats_shuff, seq, fnames_shuff

def loadImages(directory, N, H, W, dims=(32,32,3)): 
	print("Loading %d Images from %s" % (N, directory))
	X, fnames = [], []
	cats = []

	pool = multiprocessing.Pool(POOL_SIZE)

	img_names = []
	cat_files = directory + '/*/*'
	print("Loading images to disk...")
	if len(glob.glob(cat_files)) == 0:  
		print("Directory does not divide into categories.")
		for filename in sorted(glob.glob('data/*/*')): # TODO: Do something about cats.
			img_names.append(directory + os.sep + filename)
			fnames.append(filename)			
	else:
		print("Directory divides in categories.")
		num_per_cat = N / NUM_CATS
		img_names = []
		# Seems more efficient to iterate over category dirs and get exact number
		# of imgs per category. Hence, not using glob to load all imgs to disk.  
		for dirname in sorted(os.listdir(directory)):
			path = os.path.join(directory, dirname)
			filenames = [os.path.join(path, fname) for fname in sorted(os.listdir(path))]

			cats.extend([dirname] * len(filenames))
			fnames.extend(filenames)
			img_names.extend(filenames[:num_per_cat])
	
	assert(len(img_names) == N)
	imgs = pool.map(readImg, img_names)
	print("image names", img_names)
	print("fnames", fnames)

	print("Resizing images.")
	num_files = int(np.ceil(len(imgs) / POOL_SIZE))
	pairs = [(imgs[num_files * i : num_files * (i + 1)], H, W, dims) for i in range(POOL_SIZE)]
	results = pool.map(getReshapedImages, pairs)

	new_list = []
	for result in results:
		print np.shape(result)
		new_list.extend(result)

	print("Normalizing Images")
	imgs = np.array(new_list)
	imgs -= np.mean(imgs, axis = 0)
	imgs /= np.std(imgs, axis = 0)
	for img in imgs:
		img = img.astype(dtype=np.float64)
		X.append(np.array(fv.splitImage(H, W, img, dims)))

	assert(len(X) == N)
	print("Loaded %d Images from %s" % (len(X), directory))
	return np.array(X), np.array(cats), np.array(fnames)

######### TESTING ###############
data = getData(3, 3)
X_train, y_onehot_train, cats_train, seq_train, fnames_train = data['train']
X_val, y_onehot_val, cats_val, seq_val, fnames_val = data['val']
X_test, y_onehot_test, cats_test, seq_test, fnames_test = data['test']

for name, tup in data.items():
	print(name)
	X, y, cats, seq, fnames = tup
	print(X.shape, y.shape, seq.shape)
#########  DONE   ###############
