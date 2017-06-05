import gc
import numpy as np
import scipy.misc
from numpy import array
from skimage.transform import resize
JIGGLE_ROOM = 20

def fitnessScore(piece1, piece2, orientation):
    """ 
    Returns the fitness score between piece1 and
    piece2 based on the orientation.
    Orientation is either:
    - 'R': piece2 right of piece1
    - 'D': piece2 below piece1
    Assumes HWC orientation. 
    """
    h, w, _ = piece1.shape
    if orientation.lower() == "r":
        score = np.linalg.norm(piece1[:,w-1,:] - piece2[:,0,:])
    elif orientation.lower() == "d":
        score = np.linalg.norm(piece1[h-1,:,:] - piece2[0,:,:])
    else:
        raise ValueError("Unknown orientation %s" % orientation)
    return score

def splitImage(numRows, numCols, image, piece_dims=(32,32,3)):
    """
    Rescales and splits image to return numRows * numCols images
    with dim, expected_piece_dims.
    TODO: Find a cleaner way of splitting into blocks. 
    Splits along vertical and horizontal axis.
    """
    piece_height, piece_width, piece_depth = piece_dims
    # large_width, large_height, large_depth = numRows * piece_height, numCols * piece_width, piece_depth
    # # resized_img = np.array(scipy.misc.imresize(image, (large_width, large_height, large_depth), interp='nearest'))
    # resized_img = np.array(resize(image, (large_width, large_height, large_depth), 
    #                         preserve_range=True, mode='reflect')).astype(dtype=np.uint8)
    resized_img = image
    updated_pieced_dims = (piece_height + JIGGLE_ROOM, piece_width + JIGGLE_ROOM, piece_depth)
    #print(np.shape(image))
    hsplits = np.array(np.split(resized_img, numCols, axis=1))
    vsplits = np.array(np.split(hsplits, numRows, axis=1)) # Not 1 since we introduce one more dim.
    split_images = vsplits.reshape(-1, *updated_pieced_dims)
    #jiggled_imgs = []
    #for image in split_images:
    #    x_start = np.random.randint(0, JIGGLE_ROOM, 1)[0]
    #    y_start = np.random.randint(0, JIGGLE_ROOM, 1)[0]
    #    jiggled_imgs.append(image[x_start:(x_start + piece_height), y_start:(y_start + piece_width) , :])
    #gc.collect()
    return split_images #jiggled_imgs

# # TESTS
# ## Fitness Score ##
# h, w, c = 5, 4, 3
# # Two Similar Bordering R-Aligned Images.
# p1, p2 = np.ones((h, w, c)), np.ones((h, w, c))
# p1[:,w-1,:] += 3
# p2[:,0,:] += 3
# assert(fitnessScore(p1, p2, "R") == 0)

# # Two Dissimilar Bordering R-Aligned Images.
# p1, p2 = np.ones((h, w, c)), np.ones((h, w, c))
# p1[:,w-1,:] += 3
# assert(fitnessScore(p1, p2, "R") == np.sqrt(h * c * 3 * 3))

# # Two Similar Bordering D-Aligned Images.
# p1, p2 = np.ones((h, w, c)), np.ones((h, w, c))
# p1[h-1:,:,:] += 3
# p2[0,:,:] += 3
# assert(fitnessScore(p1, p2, "D") == 0)

# # Two Dissimilar Bordering D-Aligned Images.
# p1, p2 = np.ones((h, w, c)), np.ones((h, w, c))
# p1[h-1,:,:] += 3
# assert(fitnessScore(p1, p2, "D") == np.sqrt(w * c * 3 * 3))

# # Two Off-by-One Bordering R-Aligned Images.
# p1, p2 = np.ones((h, w, c)), np.ones((h, w, c))
# p1[0,w-1,0] += 3
# assert(fitnessScore(p1, p2, "R") == 3)

# # Two Off-by-One Bordering D-Aligned Images.
# p1, p2 = np.ones((h, w, c)), np.ones((h, w, c))
# p1[h-1,0,0] += 3
# assert(fitnessScore(p1, p2, "D") == 3)

# # Standard Usage Test.
# p1 = 255 * np.random.rand(32, 32, 3)
# p2 = 255 * np.random.rand(32, 32, 3)
# assert(fitnessScore(p1, p2, "R") > 0)
# assert(fitnessScore(p1, p2, "D") > 0)

# ## Split Image ##
# # TODO: Add to this.
# img = np.arange(24).reshape(4, 2, 3)
# imgs = splitImage(4, 2, img, piece_dims=(1,1,3))
# assert(imgs.shape == (8, 1, 1, 3))
# assert(np.mean(imgs) - np.mean(img) < 0.5)

# # Standard Usage
# img = 255 * np.random.rand(128, 128, 3)
# imgs = splitImage(2, 2, img, piece_dims=(32, 32, 3))
# assert(imgs.shape == (4, 32, 32, 3))
