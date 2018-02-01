from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image

from skimage.measure import block_reduce
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local, sobel
from skimage.morphology import watershed

from scipy.ndimage import gaussian_filter
from scipy.ndimage.morphology import binary_fill_holes, binary_closing, binary_erosion

from PIL import Image

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data_dir = 'E:/brains/L_slab_1/'
test_img = 'QF#HG#MR1s1#L#afdx#4266#01.tif'

downsampling_factor = 30

img = np.asarray(Image.open(data_dir + test_img), dtype='uint8')
# downsampled_img = block_reduce(img, block_size=(downsampling_factor, downsampling_factor), func=np.mean)

p2 = np.percentile(img, 2)
p98 = np.percentile(img, 98)
img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

histo = np.histogram(img, bins=np.arange(0, 256))
histo_rescaled = np.histogram(img_rescale, bins=np.arange(0, 256))

fig, axes = plt.subplots(1, 3, figsize=(15, 3))
axes[0].imshow(img, cmap=plt.cm.gray, interpolation='nearest')
axes[0].axis('off')
axes[1].imshow(img_rescale, cmap=plt.cm.gray, interpolation='nearest')
axes[1].axis('off')
axes[2].plot(histo[1][:-1], histo[0], lw=2, label='Original')
axes[2].plot(histo_rescaled[1][:-1], histo_rescaled[0], lw=2, label='Rescaled')
axes[2].legend(shadow=True)
axes[2].set_title('histogram of grey values')
plt.savefig(data_dir + 'histograms.png', bbox_inches='tight')



# print('global threshold found with otsu method:', global_thresh)
thresholded_low = (img > 25) * img
clipped = (thresholded_low < 200) * img

global_thresh = threshold_otsu(clipped)

binary_global = clipped > global_thresh
# smoothed = gaussian_filter(binary_global, 5)

before_filling = binary_global

# structure = np.ones((4, 4))

for i in range(3):
    binary_global = binary_erosion(binary_global)

for i in range(1):
    binary_global = binary_fill_holes(binary_global)

# elevation_map = sobel(smoothed)
#
# markers = np.copy(binary_global)
# segmentation = watershed(binary_global, markers)


block_size = 35
binary_adaptive = threshold_local(img, block_size, offset=10)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes

ax0.imshow(img, cmap='gray')
ax0.set_title('Image')

ax1.imshow(before_filling, cmap='gray')
ax1.set_title('Global thresholding')

ax2.imshow(binary_global, cmap='gray')
ax2.set_title('Segmentation')

for ax in axes:
    ax.axis('off')

plt.savefig(data_dir + 'thresholded.png', bbox_inches='tight')
plt.show()
plt.close()

smoothed = gaussian_filter(img_rescale, 5)   #anti-aliasing
small_img = smoothed[::downsampling_factor, ::downsampling_factor]
# mask = np.ones(small_img.shape, dtype='bool')

graph = image.img_to_graph(small_img)
graph.data = np.exp(-graph.data / graph.data.std())

print('Original image shape:', img.shape)
# print('Downsampled shape:', downsampled_img.shape)
print('Small image shape:', small_img.shape)
print('Graph shape:', graph.data.shape)

n_clusters = 2
cluster_search = range(2, 10)

plt.imshow(img_rescale, cmap='gray')
plt.axis('off')
plt.savefig(data_dir + 'rescaled.png')
plt.clf()

# for n in cluster_search:
#     print('Spectral clustering with ' + str(n) + ' clusters...')
#     labels = spectral_clustering(graph, n_clusters=n, random_state=42)
#     print('Done ' + str(n) + ' clusters.')
#     label_img = np.reshape(labels, small_img.shape)
#
#     plt.imshow(label_img, cmap='gray')
#     plt.savefig(data_dir + str(n) + '_clusters.png')
#     plt.clf()