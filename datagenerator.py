import numpy as np
import scipy.misc
import scipy.ndimage
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.transform import resize
import multiprocessing


sys.path.append('utils/')
import fitness_vectorized as fv

NUM_TEST = 10
NUM_TRAIN = 128
NUM_VAL = 128
NUM_DATA = NUM_TEST + NUM_TRAIN + NUM_VAL
DIMS=(64, 64, 3)

numRows, numCols = (3, 3)

DATA_DIR = "data/"

#for each image in that batch, the number of pieces it is cut into
 # 	making toy dataset 
# def getData(puzzle_height, puzzle_width, batch_size=700):
#   	data = {}
#   	train_correct, x_train, y_train, train_seq = [], [], [], []
#   	for i in range(batch_size):
# 		perm = np.random.permutation(puzzle_width*puzzle_height)
#  		x_train.append([[j] for j in perm])
#  		train_correct.append([[j] for j in range(puzzle_height*puzzle_width)])
#  		y = np.zeros((puzzle_width*puzzle_height, puzzle_width*puzzle_height))
#  		y[range(len(perm)), perm] = 1
#  		y_train.append(y)
# 		train_seq.append(puzzle_width*puzzle_height)
# 	train_correct, x_train, y_train, train_seq = np.array(train_correct), np.array(x_train), np.array(y_train), np.array(train_seq)
# 	x_val, y_val, val_seq = [], [], []
# 	for i in range(100):
# 		perm = np.random.permutation(puzzle_width*puzzle_height)
#  		x_val.append([[j] for j in perm])
#  		y = np.zeros((puzzle_width*puzzle_height, puzzle_width*puzzle_height))
#  		y[range(len(perm)), perm] = 1
# 		y_val.append(y)
# 		val_seq.append(puzzle_width*puzzle_height)
# 	x_val, y_val, val_seq = np.array(x_val), np.array(y_val), np.array(val_seq)

# 	data['train'] = (train_correct, x_train, y_train, train_seq)
# 	data['val'] = (x_val, y_val, val_seq)
# 	return data
 
# data = getData(2, 2, 2)
# print data['train'][0], data['train'][1], 
# perm = np.random.permutation(len(data['train'][1]))
# print 'this is ', perm
# print data['train'][0][perm],  data['train'][1][perm]

def getData(puzzle_height, puzzle_width, batch_size=-1, use_cnn=False):
	'''
	returns data : such that data['val'] = (x_val, y_val, val_seq_len)
							data['train'] = (x_train, y_train, train_seq_len)
							data['test'] = (x_test, y_test, test_seq_len) 
							# x_train = [batch1, batch2, ... batchn] - each batch should be the image
							# seq_len = [batch_size, ] # for each image in that batch, the number of pieces it is cut int
	'''
	X_flat = generateImageData(NUM_DATA, puzzle_height, puzzle_width, dims=DIMS)
	data = prepareDataset(X_flat, keep_shape = use_cnn)
	return data

def prepareDataset(X_flat, keep_shape = True):
	'''
	Splits and preprocessed dimension-formatted data into 
	train, test and validation data. 
	Returns:
	'''
	print("Preparing Dataset...")
	print(X_flat.shape)
	N, L, W, H, C = X_flat.shape
	xs = np.empty_like(X_flat)
	ys = np.zeros((N, L), dtype=np.uint8)
	ys += np.arange(L, dtype=np.uint8)

	np.random.shuffle(X_flat)
	for i in np.arange(X_flat.shape[0]):
		np.random.shuffle(ys[i])
		xs[i,:] = X_flat[i,ys[i]]

	X_train, X_val, X_test = np.split(xs, [NUM_TRAIN, NUM_TRAIN + NUM_VAL])
	y_train, y_val, y_test = np.split(ys, [NUM_TRAIN, NUM_TRAIN + NUM_VAL])
	
	if not keep_shape:
		print("Prepared Flattened Dataset!")
		X_train = X_train.reshape(NUM_TRAIN, L, -1)
		X_val = X_val.reshape(NUM_VAL, L, -1)
		print X_test.shape
		X_test = X_test.reshape(NUM_TEST, L, -1)

	# Create one-hot vectors of these arrays. 
	y_train_onehot = np.where(y_train[:,:,np.newaxis] == np.arange(L), 1, 0)
	y_val_onehot = np.where(y_val[:,:,np.newaxis] == np.arange(L), 1, 0)  
	y_test_onehot = np.where(y_test[:,:,np.newaxis] == np.arange(L), 1, 0)  

	train_seq = np.ones((len(X_train)))*L #np.full((len(X_train)), L, dtype=np.uint8)[0]
	val_seq = np.ones((len(X_val)))*L #np.full((len(X_val)), L, dtype=np.uint8)[0]
	test_seq = np.ones((len(X_test)))*L#np.full((len(X_test)), L, dtype=np.uint8)[0]

	return {
      'train': (X_train, y_train_onehot, train_seq),  
      'val'  : (X_val, y_val_onehot, val_seq), 
      'test' : (X_test, y_test_onehot, test_seq)
    }

def getReshapedImages(args):
	imgList, H, W, dims = args
	new_list = []
	for i, img in enumerate(imgList):
		large_width, large_height, large_depth = H * dims[0], W * dims[1], dims[2]
		resized_img = np.array(resize(img, (large_width, large_height, large_depth), preserve_range=True, mode='reflect'))#.astype(dtype=np.uint8)
		new_list.append(resized_img)
	return new_list

def readImg(img_name):
	return scipy.ndimage.imread(DATA_DIR + os.sep + img_name)

# need to parallelize
def generateImageData(N, H, W, dims=(32,32,3)):
	'''
	Prepares images from the data dir and returns an N*W*H*C numpy array.
	TODO: Add more transformations.   
	'''
	print("Generating Image Data...")
	imgList = []
	pool = multiprocessing.Pool(8)
	imgNames = sorted(os.listdir(DATA_DIR))[:NUM_DATA]
	imgList = pool.map(readImg, imgNames)
	print("Loaded %d images from %s." % (len(imgList), DATA_DIR))

	# print("Augmenting images by flipping.")
	# imgListFlipped = [np.fliplr(img) for img in imgList] # Expensive. 
	# print("Flipped %d images from %s." % (len(imgListFlipped), DATA_DIR))
	# imgList.extend(imgListFlipped)

	X_arr = []
	new_list = []

	num_files = int(np.ceil(len(imgList) / 8.0))
	pairs = [(imgList[num_files*i : (i + 1)*num_files],  H, W, dims) for i in range(8)]
	results = pool.map(getReshapedImages, pairs)
	for result in results:
		print np.shape(result)
		new_list.extend(result)


	imgList = new_list
	imgList = np.array(imgList)
	imgList -= np.mean(imgList, axis = 0)
	# print np.shape(imgList), np.shape(np.std(imgList, axis = 0))
	imgList /= np.std(imgList, axis = 0)
	for i, img in enumerate(imgList):
		# TODO: Check this to confirm it's good. 
		img = img.astype(dtype=np.float64)
		# np.subtract(img, np.mean(img), out=img, casting="safe")
		# np.divide(img, np.std(img), out=img, casting="safe") 
		X_arr.append(np.array(fv.splitImage(H, W, img, dims)))
	print("Generated Data!")
	return np.array(X_arr, dtype=float)

def reassemble(data, numRows, numCols):
	print("Reassembling...")
	X_train, y_train, _ = data['train']
	X_val, y_val, _ = data['val']
	X_test, y_test, _ = data['test']

	train_idx = np.random.randint(NUM_TRAIN)
	X_train0 = X_train[train_idx]
	y_train0 = y_train[train_idx]

	test_idx = np.random.randint(NUM_TEST)
	X_test0 = X_test[test_idx]
	y_test0 = y_test[test_idx]

	val_idx = np.random.randint(NUM_VAL)
	X_val0 = X_val[val_idx]
	y_val0 = y_val[val_idx]

	xs = [X_train0, X_test0, X_val0]
	ys = [y_train0, y_test0, y_val0]

	for i in np.arange(3):
		x, y = xs[i], ys[i]
		plt.figure(i)
		gs = gridspec.GridSpec(numRows, numCols)
		gs.update(wspace=0.0, hspace=0.0)
		ax = [plt.subplot(gs[i]) for i in np.arange(numRows * numCols)]

		for i in np.arange(len(x)):
			assert(sum(y[i]) == 1)
			idx = np.where(y[i] == 1)[0]
			print i, idx
			img = x[i].reshape(DIMS)
			ax[int(idx)].axis('off')
			ax[int(idx)].imshow(img)
			ax[int(idx)].set_aspect('equal')
	
	plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
	plt.show()

# TEST
# print("========= TESTING ==========")
# data = getData(numRows, numCols) 
# for n, d in data.items():
# 	print n, d.shape
# reassemble(data, numRows, numCols)
# print("========= ALL TESTS PASS =======")

