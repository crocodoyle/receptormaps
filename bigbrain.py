import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import csv
from scipy.signal import periodogram
import numpy as np

import nibabel as nib
import time



data_dir = 'E:/brains/bigbrain/'

compressed = 'bigbrain_mni152.nii.gz'
uncompressed = 'bigbrain_mni152.nii'
minc = 'bigbrain_mni152.mnc'

n_repeats = 10

if __name__ == '__main__':
    bigbrain_compressed = nib.load(data_dir + compressed)
    bigbrain_uncompressed = nib.load(data_dir + uncompressed)
    bigbrain_minc = nib.load(data_dir + minc)

    x_size, y_size, z_size = 1970, 2330, 1890

    compressed_start = time.time()
    for i in range(n_repeats):
        print('compressed slice', i)
        slice = bigbrain_compressed.dataobj[np.random.randint(x_size), ...]
        print(np.mean(slice))

    compressed_elapsed = time.time() - compressed_start

    uncompressed_start = time.time()
    for i in range(n_repeats):
        print('uncompressed slice', i)
        slice = bigbrain_compressed.dataobj[np.random.randint(x_size), ...]
        print(np.mean(slice))

    uncompressed_elapsed = time.time() - uncompressed_start

    minc_start = time.time()
    for i in range(n_repeats):
        print('minc slice', i)
        slice = bigbrain_minc.dataobj[np.random.randint(x_size), ...]
        print(np.mean(slice))

    minc_elapsed = time.time() - minc_start

    print('Compressed took:', compressed_elapsed, 'seconds')
    print('Uncompressed took:', uncompressed_elapsed, 'seconds')
    print('MINC took:', minc_elapsed, 'seconds')
