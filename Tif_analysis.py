import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset  # For custom data-sets
import glob
import tifffile as tif
from scipy.io import loadmat
import scipy.interpolate as interp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
from torch.utils.tensorboard import SummaryWriter
import pywick
import tensorflow as tf
import tensorflow.keras as keras
import argparse
from losses import *
from scipy.stats import kde
from PIL import Image
from settings import *
import mpl_scatter_density
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy import stats
from skimage import data, exposure
from skimage.exposure import match_histograms
import matplotlib.gridspec as gridspec

image1 = tif.imread('./data/timex/13_rbathy_WC_1.tiff')
image2 = np.asarray(Image.open('./snapshots/Combo_Timex.png'))
#image2 = np.asarray(Image.open('./snapshots/1487340003.Fri.Feb.17_14_00_03.GMT.2017.argus02b.cx.bright.merge.png'))
snap = np.asarray(Image.open('./snapshots/1487340000.Fri.Feb.17_14_00_00.GMT.2017.argus02b.cx.snap.merge.png'))
#image2 = cv2.rotate(image2, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
snap = cv2.rotate(snap, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
image2 = image2[304:971, :]
imagemean1 = np.mean(image1, axis=2)
imagemean2 = np.mean(image2, axis=2)

#image1 = np.where(image1>150, (101, 71, 44), image1)
imagemean1 = np.where(imagemean1>220, 75, imagemean1)

label = loadmat('./data/labels/duckgen_bathy/celeris_gen_bathy13.mat')
label = label['B']
label = cv2.resize(label, (1075, 512), interpolation=cv2.INTER_LINEAR)
label = cv2.rotate(label, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
imagemean2 = cv2.resize(imagemean2, (real_image_resize_width, real_image_resize_height), interpolation=cv2.INTER_LINEAR)
snap = cv2.resize(snap, (real_image_resize_width, real_image_resize_height), interpolation=cv2.INTER_LINEAR)
label = label[north_bound:south_bound, duckgen_cell_offset:(duckgen_cell_offset+img_cols)]
imagemean1 = imagemean1[north_bound:south_bound, duckgen_cell_offset:(duckgen_cell_offset+img_cols)]
imagemean2 = imagemean2[north_bound:south_bound, real_image_offset:(real_image_offset+img_cols)]
snap = snap[north_bound:south_bound, real_image_offset:(real_image_offset+img_cols)]
image1 = image1[304:971, :]

g_source = imagemean1
g_reference= imagemean2
g_matched = match_histograms(g_source, g_reference)
g_matched = g_matched.astype('int16')
rgb_source = image1
rgb_reference = image2
rgb_matched = match_histograms(rgb_source, rgb_reference)
rgb_matched = rgb_matched.astype('int16')
plt.subplot(2, 3, 1), plt.imshow(rgb_source, cmap='Greys_r')
plt.title("a)")
plt.ylabel("Alongshore (m)")
plt.subplot(2, 3, 2), plt.imshow(rgb_reference, cmap='Greys_r')
plt.title("b)")
plt.subplot(2, 3, 3), plt.imshow(rgb_matched, cmap='Greys_r')
plt.title("c)")
plt.subplot(2, 3, 4), plt.imshow(g_source, cmap='Greys_r')
plt.xlabel("Cross-shore (m)")
plt.ylabel("Alongshore (m)")
plt.subplot(2, 3, 5), plt.imshow(g_reference, cmap='Greys_r')
plt.xlabel("Cross-shore (m)")
plt.subplot(2, 3, 6), plt.imshow(g_matched, cmap='Greys_r')
plt.xlabel("Cross-shore (m)")

tif.imsave('./Original_1001_501.tiff', imagemean1)
tif.imsave('./Equalized_1001_501.tiff', g_matched)

fig = plt.figure(figsize=(6, 6))
grid = gridspec.GridSpec(2, 4, figure=fig)
ax0 = fig.add_subplot(grid[0, 0])
ax1 = fig.add_subplot(grid[0, 1])
ax2 = fig.add_subplot(grid[0, 2])
ax3 = fig.add_subplot(grid[0, 3])
ax4 = fig.add_subplot(grid[1, :])
norm = mpl.cm.colors.Normalize(vmax=label.max(), vmin=label.min())
cmap = mpl.cm.gist_earth

print(np.shape(label))

X = np.linspace(0, img_cols, img_cols)
Y = np.linspace(0, img_rows, img_rows)

ax0.imshow(snap[:, :])
ax0.set_ylabel('Distance Alongshore (m)', fontsize=14)
ax0.set_title('a) Argus Snapshot', fontsize=14)
ax0.set_xlabel('Cross-shore (m)', fontsize=14)
plt.tick_params(labelsize=14)
ax1.imshow(imagemean2[:, :], cmap='Greys_r')
ax1.set_title('b) Argus Timex', fontsize=14)
ax1.set_xlabel('Cross-shore (m)', fontsize=14)
plt.tick_params(labelsize=14)
ax2.imshow(g_matched[:, :], cmap='Greys_r')
ax2.set_title('c) Celeris Timex', fontsize=14)
ax2.set_xlabel('Cross-shore (m)', fontsize=14)
plt.tick_params(labelsize=14)
ax3.imshow(label[:, :], cmap='gist_earth', vmin=-6, vmax=4)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)
cbar.set_label('Elevation (m)', fontsize=14)
ax3.contour(X, Y, np.where(label > .1, 0, label), colors='white', vmin=-6, vmax=4)
ax3.set_title('d) Bathymetry', fontsize=14)
plt.tick_params(labelsize=14)
ax3.set_xlabel('Cross-shore (m)', fontsize=14)
ax4.plot(g_matched[240, :], linewidth=4, c='grey', label='Celeris')
ax4.plot(imagemean2[240, :], linewidth=4,  color='black', label='UAV')
ax4.set_title('e) 640 m Cross-shore Transect', fontsize=14)
ax4.set_ylabel('Pixel Intensity', fontsize=14)
plt.tick_params(labelsize=14)
ax4.set_xlabel('Cross-shore (m)', fontsize=14)
ax4.legend(loc=3, framealpha=1)

ax5 = ax4.twinx()
ax5.plot(label[240, :], linewidth=4,  color='cyan', label='Elevation (m)')
ax5.axhline(y=0, color='r', linestyle='-', label='Mean Water Level')
ax5.set_ylabel('Elevation (m)', fontsize=14)
plt.tick_params(labelsize=14)
ax5.legend()
plt.show()

