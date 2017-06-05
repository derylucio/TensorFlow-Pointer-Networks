import numpy as np
import glob

filename = glob.glob("*.npy")[0]
print(np.load(filename))
