from settings import *
from skimage.exposure import match_histograms
import matplotlib.gridspec as gridspec

celeris_timex = tif.imread('./data/Compare_Snaps/timex_2017-12-12T00.15.00.Z_wc2017-12-13-14.tiff')
celeris_snap = tif.imread('./data/Compare_Snaps/snap_2017-12-12T00.15.00.Z_wc2017-12-13-14.tiff')

argus_timex = np.asarray(Image.open('./data/Compare_Snaps/1513177201.Wed.Dec.13_15_00_01.GMT.2017.argus02b.cx.timex.merge.png'))
argus_snap = np.asarray(Image.open('./data/Compare_Snaps/1513177200.Wed.Dec.13_15_00_00.GMT.2017.argus02b.cx.snap.merge.png'))

label = loadmat('./data/Compare_Snaps/z2018-04-19T00.15.00.Z.mat')

name = "Dec13_15_compare_wc2018-12-13-14"

argus_timex = cv2.rotate(argus_timex, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
argus_snap = cv2.rotate(argus_snap, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)

celeris_timex_mean = np.mean(celeris_timex, axis=2)
celeris_snap_mean = np.mean(celeris_snap, axis=2)


argus_timex_mean = np.mean(argus_timex, axis=2)
argus_snap_mean = np.mean(argus_snap, axis=2)
argus_timex_mean = cv2.resize(argus_timex_mean, (real_image_resize_width+100, real_image_resize_height), interpolation=cv2.INTER_LINEAR)
argus_snap_mean = cv2.resize(argus_snap_mean, (real_image_resize_width+100, real_image_resize_height), interpolation=cv2.INTER_LINEAR)

label = label['B']
label = cv2.rotate(label, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
label = cv2.resize(label, (real_image_resize_width+100, real_image_resize_height), interpolation=cv2.INTER_LINEAR)

print(np.shape(label))

north_bound = 300
south_bound = north_bound+512

celeris_timex_mean = celeris_timex_mean[north_bound:south_bound, :img_cols]
celeris_snap_mean = celeris_snap_mean[north_bound:south_bound, :img_cols]

argus_timex_mean = argus_timex_mean[north_bound:south_bound, 100:(100+img_cols)]
argus_snap_mean = argus_snap_mean[north_bound:south_bound, 100:(100+img_cols)]

label = label[north_bound:south_bound, 50:(img_cols)]

g_source = celeris_timex_mean
g_snap = celeris_snap_mean
g_reference= argus_timex_mean[:, :450]
g_matched = match_histograms(g_source, g_reference)
snap_matched = match_histograms(g_snap, g_reference)
g_matched = g_matched.astype('int16')
snap_matched = snap_matched.astype('int16')
rgb_source = celeris_timex
rgb_reference = argus_timex
rgb_matched = match_histograms(rgb_source, rgb_reference)
rgb_matched = rgb_matched.astype('int16')

"""plt.subplot(2, 3, 1), plt.imshow(rgb_source, cmap='Greys_r')
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

tif.imsave('./Original_1001_501.tiff', celeris_timex_mean)
tif.imsave('./Equalized_1001_501.tiff', g_matched)"""

fig = plt.figure(figsize=(16, 9))

norm = mpl.cm.colors.Normalize(vmax=1, vmin=-6)
cmap = mpl.cm.gist_earth

grid = gridspec.GridSpec(2, 4, figure=fig)
ax0 = fig.add_subplot(grid[0, 0])
ax1 = fig.add_subplot(grid[0, 1])
ax2 = fig.add_subplot(grid[0, 2])
ax3 = fig.add_subplot(grid[0, 3])
ax4 = fig.add_subplot(grid[1, :])

X = np.linspace(0, img_cols, img_cols)
Y = np.linspace(0, img_rows, img_rows)

ax0.imshow(snap_matched, cmap='Greys_r')
ax0.set_ylabel('Distance Alongshore (m)', fontsize=14)
ax0.set_title('a) Celeris Snapshot', fontsize=14)
ax0.set_xlabel('Cross-shore (m)', fontsize=14)
plt.tick_params(labelsize=14)

ax1.imshow(argus_snap_mean, cmap='Greys_r')
ax1.set_title('b) Argus Snapshot', fontsize=14)
ax1.set_xlabel('Cross-shore (m)', fontsize=14)
plt.tick_params(labelsize=14)

ax2.imshow(g_matched, cmap='Greys_r')
ax2.set_title('c) Celeris Timex', fontsize=14)
ax2.set_xlabel('Cross-shore (m)', fontsize=14)
plt.tick_params(labelsize=14)

"""ax3.imshow(label[:, :], cmap='gist_earth', vmin=-6, vmax=4)
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)
cbar.set_label('Elevation (m)', fontsize=14)
ax3.contour(X, Y, np.where(label > .1, 0, label), colors='white', vmin=-6, vmax=4)
ax3.set_title('d) Bathymetry', fontsize=14)
plt.tick_params(labelsize=14)
ax3.set_xlabel('Cross-shore (m)', fontsize=14)"""

ax3.imshow(argus_timex_mean, cmap='Greys_r')
ax3.set_title('d) Argus Timex', fontsize=14)
ax3.set_xlabel('Cross-shore (m)', fontsize=14)
plt.tick_params(labelsize=14)

ax4.plot(np.mean(g_matched[:, :450],axis=0), linewidth=4, c='grey', label='Celeris')
ax4.plot(np.mean(argus_timex_mean[:, :450],axis=0), linewidth=4,  color='black', label='Argus')
ax4.set_title('e) Average Cross-shore Transect (Timex)', fontsize=14)
ax4.set_ylabel('Pixel Intensity', fontsize=14)
plt.tick_params(labelsize=14)
ax4.set_xlabel('Cross-shore (m)', fontsize=14)
ax4.legend(loc=3, framealpha=1)

ax5 = ax4.twinx()
ax5.plot(np.mean(label[:, :450],axis=0), linewidth=4,  color='cyan', label='Elevation (m)')
ax5.axhline(y=0, color='r', linestyle='-', label='Mean Water Level')
ax5.set_ylabel('Elevation (m)', fontsize=14)
plt.tick_params(labelsize=14)
ax5.legend(loc=1, framealpha=1)

fig.canvas.draw()  # draw the canvas, cache the renderer
image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
plt.close('all')

tif.imsave('./data/Compare_Snaps/' + name + '.tiff', image)

