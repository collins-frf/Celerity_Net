# -*- coding:utf-8 -*-
#all the imports for every file
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch, SqrtStretch, MinMaxInterval, PercentileInterval, ManualInterval
from settings import *
from skimage.exposure import match_histograms
import mpl_scatter_density
import argparse
import cv2
import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import scipy.interpolate as interp

argus_timex = np.asarray(Image.open('./data/Compare_Snaps/1513177201.Wed.Dec.13_15_00_01.GMT.2017.argus02b.cx.timex.merge.png'))
argus_timex_mean = np.mean(argus_timex, axis=2)
argus_timex_mean = cv2.resize(argus_timex_mean, (real_image_resize_width + 100, real_image_resize_height),
                             interpolation=cv2.INTER_LINEAR)
north_bound = 550
south_bound = north_bound+512
argus_snap_mean = argus_timex_mean[north_bound:south_bound, 100:(100 + img_cols)]
g_reference = argus_timex_mean[:, :450]


def plot_for_gif(img_mean, snap_mean, label_mean, pred_mean, diff_mean, i, pred_list, uncertainty_2d):

    X = np.linspace(zeroline, crosshore_distance_meters, crosshore_distance_meters-zeroline)
    Y = np.linspace(0, alongshore_distance_meters, alongshore_distance_meters)

    label_transect = np.mean(label_mean[:, :, i], axis=0)
    nrms_transect = np.power(np.sum(np.power(diff_mean[:, :, i]/(np.where((-label_mean[:, :, i] > .3), label_mean[:, :, i], 1)), 2), axis=0) / (bathy_cols-downsample_zeroline), .5)
    rms_transect = np.power(np.sum(np.power(diff_mean[:, :, i], 2), axis=0) / (bathy_cols - downsample_zeroline), .5)

    label_up = cv2.resize(label_mean[:, :, i],
                          (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)
    pred_up = cv2.resize(pred_mean[:, :, i],
                         (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)


    cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m", "-3m", "-2.5m", "-2m",
                 "-1.5m", "-1m", "-.5m", "0m"]
    fmt = {}
    for l, s in zip([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], cs_labels):
        fmt[l] = s

    fig = plt.figure(figsize=(16, 9))
    grid = gridspec.GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[1, 2])

    norm = mpl.cm.colors.Normalize(vmax=1, vmin=-6)
    cmap = mpl.cm.gist_earth

    img_norm = mpl.cm.colors.Normalize(vmax=.9, vmin=.1)
    img_cmap = mpl.cm.Greys_r

    if (snap) and (not snap_only):
        img_mean = (img_mean + snap_mean)/2
    ax0.imshow(img_mean[:, downsample_zeroline:, i], cmap='Greys_r',
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters], vmax=1, vmin=0)
    cs = ax0.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-4, vmax=2, alpha=.5,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax0.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=img_norm, cmap=img_cmap), ax=ax0)
    cbar.set_label('Pixel Intensity', fontsize=14)
    ax0.set_title('a) Averaged Input', fontsize=16)
    ax0.set_ylabel('Alongshore (m)', fontsize=14)
    plt.tick_params(labelsize=14)

    ax1.imshow(diff_mean[:, downsample_zeroline:, i], cmap='bwr', vmin=-.5, vmax=.5,
               extent=[zeroline, crosshore_distance_meters, 0,
                       alongshore_distance_meters])
    cs = ax1.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                     vmin=-4, vmax=2, alpha=.5,
                     colors='black',
                     levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                     linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                     linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax1.clabel(cs, fmt=fmt,
               inline_spacing=2, fontsize='x-small', )
    ax1.set_title('b) Difference', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=.5, vmin=-.5), cmap=mpl.cm.bwr), ax=ax1)
    cbar.set_label('(m)', fontsize=14)

    ax2.imshow(uncertainty_2d[:, :, i], cmap='inferno', vmin=0, vmax=1,
               extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    ax2.set_title('c) Sensitivity', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=1, vmin=0), cmap=mpl.cm.inferno), ax=ax2)
    cbar.set_label('(m)', fontsize=14)

    ax3.imshow(label_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax3)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax3.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax3.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax3.set_title('d) Truth', fontsize=16)
    ax3.set_ylabel('Alongshore (m)', fontsize=14)
    ax3.set_xlabel('Cross-shore (m)', fontsize=14)

    ax4.imshow(pred_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    ax4.set_title('e) Predicted', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax4)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax4.contour(X, Y, np.where(pred_up[:, :img_cols] > .1, 0, np.flip(pred_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax4.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax4.set_xlabel('Cross-shore (m)', fontsize=14)

    pred_transects = [[] for j in range(2*ensemble_runs)]
    for j in range(2*ensemble_runs):
        pred_transects[j] = np.mean(pred_list[:, downsample_zeroline:, i, j],axis=0)
    x = np.linspace(zeroline, crosshore_distance_meters, num=len(label_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    label_interpolate = interp.InterpolatedUnivariateSpline(x, label_transect)

    for j in range(len(pred_transects) - 1):
        pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[j])
        ax5.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey')

    pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[-1])
    ax5.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey',
                   label='Indiv Preds')
    pred_interpolate = interp.InterpolatedUnivariateSpline(x, np.mean(pred_transects, axis=0))
    ax5.plot(x_new, pred_interpolate(x_new), c='red', label='Mean Pred')
    ax5.plot(np.linspace(zeroline, crosshore_distance_meters), label_interpolate(x_new), c='cyan', label='Truth')
    ax5.set_ylabel('Elevation (m)', fontsize=14)
    plt.tick_params(labelsize=14)
    ax5.set_xlabel('Cross-shore (m)', fontsize=14)
    ax5.set_ylim(ymax=1, ymin=-8)

    x = np.linspace(zeroline, crosshore_distance_meters, num=len(rms_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    interpolate = interp.InterpolatedUnivariateSpline(x, rms_transect)
    ax6 = ax5.twinx()
    ax6.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='b', label='RMSE')
    interpolate = interp.InterpolatedUnivariateSpline(x, nrms_transect)
    ax6.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='black', label='Depth Norm RMSE')
    plt.title('f) Alongshore Average Transects', fontsize=16)
    ax6.legend(loc=1)
    ax5.legend(loc=6)

    ax6.set_ylabel('', fontsize=14)
    ax6.set_ylim(ymin=0, ymax=1)

    fig.tight_layout(pad=3)
    #plt.subplots_adjust(right=.6)
    # Used to return the plot as an image array
    #fig.canvas.draw()       # draw the canvas, cache the renderer
    #image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #plt.close('all')
    #return image


def plot_for_gif2(img_mean, snap_mean, label_mean, pred_mean, diff_mean, i, pred_list, uncertainty_2d):

    X = np.linspace(zeroline, crosshore_distance_meters, crosshore_distance_meters-zeroline)
    Y = np.linspace(0, alongshore_distance_meters, alongshore_distance_meters)

    label_up = cv2.resize(label_mean[:, :, i],
                          (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)
    pred_up = cv2.resize(pred_mean[:, :, i],
                         (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)

    img_mean[:, :, i] = match_histograms(img_mean[:, :, i], g_reference)
    snap_mean[:, :, i] = match_histograms(snap_mean[:, :, i], g_reference)
    snap_mean = snap_mean.astype('int16')
    img_mean = img_mean.astype('int16')

    cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m", "-3m", "-2.5m", "-2m",
                 "-1.5m", "-1m", "-.5m", "0m"]
    fmt = {}
    for l, s in zip([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], cs_labels):
        fmt[l] = s

    fig = plt.figure(figsize=(16, 9))
    grid = gridspec.GridSpec(2, 3, figure=fig)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 0])
    ax4 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[1, 2])
    print(i)
    norm = mpl.cm.colors.Normalize(vmax=1, vmin=-6)
    cmap = mpl.cm.gist_earth

    img_cmap = mpl.cm.Greys_r
    img_norm = ImageNormalize(img_mean, interval=ManualInterval(vmin=0, vmax=255))

    ax0.imshow(img_mean[:, :, i], cmap='Greys_r', norm=img_norm,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])

    cs = ax0.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-4, vmax=2, alpha=.5,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax0.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=img_norm, cmap=img_cmap), ax=ax0)
    cbar.set_label('Pixel Intensity', fontsize=14)
    ax0.set_title('a) Timex Input', fontsize=16)
    ax0.set_ylabel('Alongshore (m)', fontsize=14)

    ax1.imshow(label_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax1.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax1.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax1.set_title('b) Truth', fontsize=16)



    ax2.imshow(uncertainty_2d[:, :, i], cmap='inferno', vmin=0, vmax=1,
               extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    ax2.set_title('c) Uncertainty', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=1, vmin=0), cmap=mpl.cm.inferno), ax=ax2)
    cbar.set_label('(m)', fontsize=14)
    cs = ax2.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                     vmin=-4, vmax=2, alpha=.5,
                     colors='white',
                     levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                     linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                     linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax2.clabel(cs, fmt=fmt,
               inline_spacing=2, fontsize='x-small', )

    ax3.imshow(snap_mean[:, downsample_zeroline:, i], cmap='Greys_r', norm=img_norm,
               extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    cs = ax3.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-4,
                     vmax=2, alpha=.5,
                     colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                             'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                     levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                     linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                     linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax3.clabel(cs, fmt=fmt,
               inline_spacing=2, fontsize='x-small', )
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=img_norm, cmap=img_cmap), ax=ax3)
    cbar.set_label('Pixel Intensity', fontsize=14)
    ax3.set_title('d) Snap Input', fontsize=16)
    ax3.set_xlabel('Cross-shore (m)', fontsize=14)
    ax3.set_ylabel('Alongshore (m)', fontsize=14)

    ax4.imshow(pred_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    ax4.set_title('e) Predicted', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax4)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax4.contour(X, Y, np.where(pred_up[:, :img_cols] > .1, 0, np.flip(pred_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax4.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax4.set_xlabel('Cross-shore (m)', fontsize=14)

    ax5.imshow(np.abs(diff_mean[:, downsample_zeroline:, i]), cmap='inferno', vmin=0, vmax=1,
               extent=[zeroline, crosshore_distance_meters, 0,
                       alongshore_distance_meters])
    cs = ax5.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                     vmin=-4, vmax=2, alpha=.5,
                     colors='white',
                     levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                     linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                     linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax5.clabel(cs, fmt=fmt,
               inline_spacing=2, fontsize='x-small', )
    ax5.set_title('f) Absolute Error', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=1, vmin=0), cmap=mpl.cm.inferno), ax=ax5)
    cbar.set_label('(m)', fontsize=14)
    ax5.set_xlabel('Cross-shore (m)', fontsize=14)

    fig.tight_layout(pad=3)
    #plt.subplots_adjust(right=.6)
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #plt.savefig(("./results/plot/results" + str(i) + ".png"))
    plt.close('all')
    return image


def plot_for_paper(img_mean, label_mean, pred_mean, diff_mean, pred_list,
                   img_mean2, pred_mean2, diff_mean2, pred_list2,
                    img_mean3, pred_mean3, diff_mean3, pred_list3, i):

    img_mean3[:, :, i] = match_histograms(img_mean3[:, :, i], g_reference)
    img_mean2[:, :, i] = match_histograms(img_mean2[:, :, i], g_reference)
    img_mean2 = img_mean2.astype('int16')
    img_mean3 = img_mean3.astype('int16')


    X = np.linspace(zeroline, crosshore_distance_meters, crosshore_distance_meters-zeroline)
    Y = np.linspace(0, alongshore_distance_meters, alongshore_distance_meters)

    #label_transect = np.mean(label_mean[:, :, i], axis=0)
    label_transect = label_mean[50, :, i]
    nrms_transect3 = np.power(np.sum(np.power(diff_mean3[:, :, i], 2), axis=0)/np.sum(np.power(np.where(label_mean[:, :, i] < -.1, label_mean[:, :, i], 1), 2), axis=0), .5)
    rms_transect3 = np.power(np.sum(np.power(diff_mean3[:, :, i], 2), axis=0) / (bathy_cols), .5)
    nrms_transect2 = np.power(np.sum(np.power(diff_mean2[:, :, i], 2), axis=0)/np.sum(np.power(np.where(label_mean[:, :, i] < -.1, label_mean[:, :, i], 1), 2), axis=0), .5)
    rms_transect2 = np.power(np.sum(np.power(diff_mean2[:, :, i], 2), axis=0) / (bathy_cols), .5)
    nrms_transect = np.power(np.sum(np.power(diff_mean[:, :, i], 2), axis=0)/np.sum(np.power(np.where(label_mean[:, :, i] < -.1, label_mean[:, :, i], 1), 2), axis=0), .5)
    rms_transect = np.power(np.sum(np.power(diff_mean[:, :, i], 2), axis=0) / (bathy_cols), .5)

    pred_up3 = cv2.resize(pred_mean3[:, :, i],
                         (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)
    pred_up2 = cv2.resize(pred_mean2[:, :, i],
                         (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)
    pred_up = cv2.resize(pred_mean[:, :, i],
                          (crosshore_distance_meters - zeroline, alongshore_distance_meters),
                          interpolation=cv2.INTER_CUBIC)
    label_up = cv2.resize(label_mean[:, :, i],
                          (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)

    cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m", "-3m", "-2.5m", "-2m",
                 "-1.5m", "-1m", "-.5m", "0m"]
    fmt = {}
    for l, s in zip([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], cs_labels):
        fmt[l] = s

    fig = plt.figure(figsize=(16, 9))
    grid = gridspec.GridSpec(3, 4, figure=fig)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[0, 3])
    ax5 = fig.add_subplot(grid[1, 0])
    ax6 = fig.add_subplot(grid[1, 1])
    ax7 = fig.add_subplot(grid[1, 2])
    ax8 = fig.add_subplot(grid[1, 3])
    ax10 = fig.add_subplot(grid[2, 0])
    ax11 = fig.add_subplot(grid[2, 1])
    ax12 = fig.add_subplot(grid[2, 2])
    ax13 = fig.add_subplot(grid[2, 3])

    norm = mpl.cm.colors.Normalize(vmax=1, vmin=-6)
    cmap = mpl.cm.gist_earth

    img_norm = ImageNormalize(img_mean, interval=ManualInterval(vmin=0, vmax=255))
    img_cmap = mpl.cm.Greys_r

    ax0.imshow(img_mean3[:, downsample_zeroline:, i], cmap='Greys_r', norm=img_norm,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    cs = ax0.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-4, vmax=2, alpha=.5,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax0.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=img_norm, cmap=img_cmap), ax=ax0)
    cbar.set_label('Pixel Intensity', fontsize=14)
    ax0.set_title('a) Timex', fontsize=16)
    ax0.set_ylabel('Alongshore (m)', fontsize=14)

    ax1.imshow(pred_mean3[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    ax1.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    ax1.set_title('b) Predicted', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax1.contour(X, Y, np.where(pred_up3[:, :img_cols] > .1, 0, np.flip(pred_up3[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax1.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )

    ax2.imshow(diff_mean3[:, downsample_zeroline:, i], cmap='bwr', vmin=-2, vmax=2,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    cs = ax2.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=.5,
                        colors='black',
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax2.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    #ax4 = fig.add_subplot(2, 3, 2), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    ax2.set_title('c) Difference', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=.5, vmin=-.5), cmap=mpl.cm.bwr), ax=ax2)
    cbar.set_label('(m)', fontsize=14)

    pred_transects = [[] for j in range(2 * ensemble_runs)]
    for j in range(2 * ensemble_runs):
        #pred_transects[j] = np.mean(pred_list3[:, downsample_zeroline:, i, j], axis=0)
        pred_transects[j] = pred_list3[50, downsample_zeroline:, i, j]
    x = np.linspace(zeroline, crosshore_distance_meters, num=len(label_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    label_interpolate = interp.InterpolatedUnivariateSpline(x, label_transect)

    for j in range(len(pred_transects) - 1):
        pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[j])
        ax3.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey')

    pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[-1])

    ax3.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey',
             label='Indiv Preds')
    pred_interpolate = interp.InterpolatedUnivariateSpline(x, np.mean(pred_transects, axis=0))
    ax3.plot(x_new, pred_interpolate(x_new), c='red', label='Mean Pred')
    ax3.plot(np.linspace(zeroline, crosshore_distance_meters), label_interpolate(x_new), c='cyan', label='Truth')
    ax3.set_ylabel('Elevation (m)', fontsize=14)
    ax3.set_ylim(ymax=1, ymin=-8)

    x = np.linspace(zeroline, crosshore_distance_meters, num=len(rms_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    interpolate = interp.InterpolatedUnivariateSpline(x, rms_transect3)
    ax4 = ax3.twinx()
    ax4.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='b', label='RMSE')
    interpolate = interp.InterpolatedUnivariateSpline(x, nrms_transect3)
    ax4.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='black', label='Depth Norm RMSE')
    #plt.title('d) Alongshore Average Transects', fontsize=16)
    plt.title('d) Cross-shore Transect', fontsize=16)
    #ax4.legend(loc=1)
    #ax3.legend(loc=6)
    ax4.set_ylabel('', fontsize=14)
    ax4.set_ylim(ymin=0, ymax=2)

    ax5.imshow(img_mean2[:, downsample_zeroline:, i], cmap='Greys_r', norm=img_norm,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])

    cs = ax5.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-4, vmax=2, alpha=.5,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax5.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=img_norm, cmap=img_cmap), ax=ax5)
    cbar.set_label('Pixel Intensity', fontsize=14)
    ax5.set_title('e) Snap', fontsize=16)
    ax5.set_ylabel('Alongshore (m)', fontsize=14)

    ax6.imshow(pred_mean2[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    ax6.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    ax6.set_title('f) Predicted', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax6)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax6.contour(X, Y, np.where(pred_up2[:, :img_cols] > .1, 0, np.flip(pred_up2[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax6.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )

    ax7.imshow(diff_mean2[:, downsample_zeroline:, i], cmap='bwr', vmin=-2, vmax=2,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    cs = ax7.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=.5,
                        colors='black',
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax7.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )

    ax7.set_title('g) Difference', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=.5, vmin=-.5), cmap=mpl.cm.bwr), ax=ax7)
    cbar.set_label('(m)', fontsize=14)

    pred_transects = [[] for j in range(2 * ensemble_runs)]
    for j in range(2 * ensemble_runs):
        #pred_transects[j] = np.mean(pred_list2[:, downsample_zeroline:, i, j], axis=0)
        pred_transects[j] = pred_list2[50, downsample_zeroline:, i, j]
    x = np.linspace(zeroline, crosshore_distance_meters, num=len(label_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    label_interpolate = interp.InterpolatedUnivariateSpline(x, label_transect)

    for j in range(len(pred_transects) - 1):
        pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[j])
        ax8.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey')

    pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[-1])

    ax3.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey',
             label='Indiv Preds')
    pred_interpolate = interp.InterpolatedUnivariateSpline(x, np.mean(pred_transects, axis=0))
    ax8.plot(x_new, pred_interpolate(x_new), c='red', label='Mean Pred')
    ax8.plot(np.linspace(zeroline, crosshore_distance_meters), label_interpolate(x_new), c='cyan', label='Truth')
    ax8.set_ylabel('Elevation (m)', fontsize=14)
    ax8.set_ylim(ymax=1, ymin=-8)

    x = np.linspace(zeroline, crosshore_distance_meters, num=len(rms_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    interpolate = interp.InterpolatedUnivariateSpline(x, rms_transect2)
    ax9 = ax8.twinx()
    ax9.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='b', label='RMSE')
    interpolate = interp.InterpolatedUnivariateSpline(x, nrms_transect2)
    ax9.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='black', label='Depth Norm RMSE')
    #plt.title('h) Alongshore Average Transects', fontsize=16)
    plt.title('h) Cross-shore Transect', fontsize=16)
    #ax9.legend(loc=1)
    #ax8.legend(loc=6)
    ax9.set_ylabel('', fontsize=14)
    ax9.set_ylim(ymin=0, ymax=2)

    ax10.imshow(label_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    ax10.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax10)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax10.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax10.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax10.set_title('i) Truth', fontsize=16)
    ax10.set_ylabel('Alongshore (m)', fontsize=14)
    ax10.set_xlabel('Cross-shore (m)', fontsize=14)

    ax11.imshow(pred_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    ax11.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    ax11.set_title('j) Predicted', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax11)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax11.contour(X, Y, np.where(pred_up[:, :img_cols] > .1, 0, np.flip(pred_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax11.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax11.set_xlabel('Cross-shore (m)', fontsize=14)

    ax12.imshow(diff_mean[:, downsample_zeroline:, i], cmap='bwr', vmin=-2, vmax=2,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    cs = ax12.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=.5,
                        colors='black',
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax12.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )

    ax12.set_xlabel('Cross-shore (m)', fontsize=14)
    ax12.set_title('k) Difference', fontsize=16)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=.5, vmin=-.5), cmap=mpl.cm.bwr), ax=ax12)
    cbar.set_label('(m)', fontsize=14)

    pred_transects = [[] for j in range(2 * ensemble_runs)]
    for j in range(2 * ensemble_runs):
        #pred_transects[j] = np.mean(pred_list[:, downsample_zeroline:, i, j], axis=0)
        pred_transects[j] = pred_list[50, downsample_zeroline:, i, j]
    x = np.linspace(zeroline, crosshore_distance_meters, num=len(label_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    label_interpolate = interp.InterpolatedUnivariateSpline(x, label_transect)

    for j in range(len(pred_transects) - 1):
        pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[j])
        ax13.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey')

    pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[-1])

    ax13.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey',
             label='Indiv Preds')
    pred_interpolate = interp.InterpolatedUnivariateSpline(x, np.mean(pred_transects, axis=0))
    ax13.plot(x_new, pred_interpolate(x_new), c='red', label='Mean Pred')
    ax13.plot(np.linspace(zeroline, crosshore_distance_meters), label_interpolate(x_new), c='cyan', label='Truth')
    ax13.set_ylabel('Elevation (m)', fontsize=14)
    ax13.set_xlabel('Cross-shore (m)', fontsize=14)
    ax13.set_ylim(ymax=1, ymin=-8)

    x = np.linspace(zeroline, crosshore_distance_meters, num=len(rms_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    interpolate = interp.InterpolatedUnivariateSpline(x, rms_transect)
    ax14 = ax13.twinx()
    ax14.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='b', label='RMSE')
    interpolate = interp.InterpolatedUnivariateSpline(x, nrms_transect)
    ax14.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='black', label='Depth Norm RMSE')
    #plt.title('l) Alongshore Average Transects', fontsize=16)
    plt.title('l) Cross-shore Transect', fontsize=16)
    #ax14.legend(loc=1)
    #ax13.legend(loc=6)
    ax14.set_ylabel('', fontsize=14)
    ax14.set_ylim(ymin=0, ymax=2)

    plt.subplots_adjust(top=.97, wspace=.52, hspace=.38, bottom=.06, right=.97, left=.09)
    #plt.show()
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #plt.savefig(("./results/plot/comp" + str(i) + ".png"))
    plt.close('all')
    return image


def make_boxplot(bp):
    colors = ['pink', "lightblue", "lightgreen"]
    i=0
    for box in bp['boxes']:
        # change outline color
        box.set(color=colors[i], linewidth=2)
        i+=1
        if i == 3:
            i = 0
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    # change color and linewidth of the whiskers
    for whisker in bp['whiskers']:
        whisker.set(color='blue', linewidth=2)
    # change color and linewidth of the caps
    for cap in bp['caps']:
        cap.set(color='blue', linewidth=2)
    medians = []
    # change color and linewidth of the medians
    for median in bp['medians']:
        median.set(color='green', linewidth=2)
        medianY = []
        for j in range(2):
            medianY.append(median.get_ydata()[j])
        medians.append(medianY[0])
    # change the style of fliers and their fill
    for flier in bp['fliers']:
        flier.set(marker='o', color='red', alpha=0.3)
    return bp, medians



def plot(img_mean, snap_mean, label_mean, pred_mean, diff_mean, rms2d_mean, mae2d_mean,
                 uncertainty_2d, diff_histo, rms_histo, pred_list, wc_list, within_2d, within_2d_mean,
         img_mean2, snap_mean2, label_mean2, pred_mean2, diff_mean2, rms2d_mean2, mae2d_mean2,
         uncertainty_2d2, diff_histo2, rms_histo2, pred_list2, within_2d2, within_2d_mean2,
         img_mean3, snap_mean3, label_mean3, pred_mean3, diff_mean3, rms2d_mean3, mae2d_mean3,
         uncertainty_2d3, diff_histo3, rms_histo3, pred_list3, within_2d3, within_2d_mean3,
         mae1d, diff1d, within1d, U_bin, U_mae, H_bin, H_rms, H_bias, D_bin, D_rms, D_bias, F_bin, F_rms, F_bias, mae1d2,
         diff1d2, within1d2, U_bin2, U_mae2, H_bin2, H_rms2, H_bias2, D_bin2, D_rms2, D_bias2, F_bin2, F_rms2, F_bias2, mae1d3,
         diff1d3, within1d3, U_bin3, U_mae3, H_bin3, H_rms3, H_bias3, D_bin3, D_rms3, D_bias3, F_bin3, F_rms3, F_bias3):

    if args.fullstats:
        mpl.rcParams['agg.path.chunksize'] = zeroline

        img1d = img_mean[:, downsample_zeroline:].flatten()
        pred1d = pred_mean.flatten()
        label1d = label_mean.flatten()
        uc1d = uncertainty_2d.flatten()
        new_uncertainty_2d = uncertainty_2d
        new_uncertainty_2d = np.float32(new_uncertainty_2d)
        b_uncertainty_2d = new_uncertainty_2d
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, .001)
        for i in range(len(uncertainty_2d[:, :])):
            _, label, centers = cv2.kmeans(new_uncertainty_2d[:, :, i], 10, None, criteria, 100, cv2.KMEANS_PP_CENTERS)
            b_uncertainty_2d[:, :, i] = centers[label.flatten()]
            #fig = plt.figure()
            #grid = gridspec.GridSpec(1, 2, figure=fig)
            #ax0 = fig.add_subplot(grid[0, 0])
            #ax1 = fig.add_subplot(grid[0, 1])
            #ax0.imshow(new_uncertainty_2d[:, :, i])
            #ax1.imshow(centers[label.flatten()])
            #plt.show()

        # figure 1
        fig = plt.figure()
        grid = gridspec.GridSpec(1, 1, figure=fig)
        ax0 = fig.add_subplot(grid[:, :], projection='scatter_density')

        norm = ImageNormalize(vmin=0, vmax=1000, stretch=LogStretch())
        cmap = mpl.cm.jet
        density = ax0.scatter_density(-pred1d, -label1d, norm=norm, cmap='jet')
        fig.colorbar(density, ax=ax0).set_label(label='Number of Points per Pixel', size=14)
        ax0.plot([0, 8], [0, 8], c='black')
        ax0.set_ylim(0, 8)
        ax0.set_xlabel('Prediction Depth (m)', fontsize=16)
        ax0.set_ylabel('Truth Depth (m)', fontsize=16)

        # figure 2
        fig = plt.figure()
        grid = gridspec.GridSpec(4, 3, figure=fig)
        ax0 = fig.add_subplot(grid[0, 0])
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[2, 0])
        ax3 = fig.add_subplot(grid[3, 0])
        ax4 = fig.add_subplot(grid[0, 1])
        ax5 = fig.add_subplot(grid[1, 1])
        ax6 = fig.add_subplot(grid[2, 1])
        ax7 = fig.add_subplot(grid[3, 1])
        ax8 = fig.add_subplot(grid[0, 2])
        ax9 = fig.add_subplot(grid[1, 2])
        ax10 = fig.add_subplot(grid[2, 2])
        ax11 = fig.add_subplot(grid[3, 2])

        norm = mpl.cm.colors.Normalize(vmax=1.0, vmin=0)
        ax0.imshow(rms2d_mean3, cmap='jet', vmax=1.0, vmin=0, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax0.set_title('Timex \na) RMSE', fontsize=16)
        #ax0.set_ylabel('Alongshore (m)', fontsize=16)
        ax0.set_anchor('W')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax0)
        cbar.set_label('(m)', fontsize=12)

        norm = mpl.cm.colors.Normalize(vmax=.25, vmin=-.25)
        ax1.imshow(np.mean(diff_mean3, axis=-1), cmap='bwr', vmin=-.25, vmax=.25, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax1.set_title('d) Bias', fontsize=16)
        ax1.set_xlabel('Cross-Shore (m)', fontsize=12)
        ax1.set_anchor('W')
        #ax1.set_ylabel('Alongshore (m)', fontsize=16)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=ax1)
        cbar.set_label('(m)', fontsize=12)

        ax2.hist(np.where(rms_histo3 < 1.5, rms_histo3, .5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax2.set_title('h) RMSE Histogram', fontsize=16)
        ax2.set_xlabel('RMSE (m)', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_ylim(ymax=140)
        ax2.set_xlim(xmax=1.4)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        ax3.hist(np.where(diff_histo3 > -1.5, diff_histo3, -.5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax3.set_title('k) Bias Histogram', fontsize=16)
        ax3.set_xlabel('Bias (m)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_ylim(ymax=200)
        ax3.set_xlim(xmax=1.5, xmin=-1.5)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")

        norm = mpl.cm.colors.Normalize(vmax=1.0, vmin=0)
        ax4.imshow(rms2d_mean2, cmap='jet', vmax=1.0, vmin=0, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax4.set_title('Snap \nb) RMSE', fontsize=16)
        #ax0.set_ylabel('Alongshore (m)', fontsize=16)
        ax4.set_anchor('W')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax4)
        cbar.set_label('(m)', fontsize=12)

        norm = mpl.cm.colors.Normalize(vmax=.25, vmin=-.25)
        ax5.imshow(np.mean(diff_mean2, axis=-1), cmap='bwr', vmin=-.25, vmax=.25, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax5.set_title('e) Bias', fontsize=16)
        ax5.set_xlabel('Cross-Shore (m)', fontsize=12)
        ax5.set_anchor('W')
        #ax1.set_ylabel('Alongshore (m)', fontsize=16)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=ax5)
        cbar.set_label('(m)', fontsize=12)

        ax6.hist(np.where(rms_histo2 < 1.5, rms_histo2, .5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax6.set_title('i) RMSE Histogram', fontsize=16)
        ax6.set_xlabel('RMSE (m)', fontsize=12)
        ax6.set_ylabel('Count', fontsize=12)
        ax6.set_ylim(ymax=140)
        ax6.set_xlim(xmax=1.4)
        ax6.yaxis.tick_right()
        ax6.yaxis.set_label_position("right")

        ax7.hist(np.where(diff_histo2 > -1.5, diff_histo2, -.5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax7.set_title('l) Bias Histogram', fontsize=16)
        ax7.set_xlabel('Bias (m)', fontsize=12)
        ax7.set_ylabel('Count', fontsize=12)
        ax7.set_ylim(ymax=200)
        ax7.set_xlim(xmax=1.5, xmin=-1.5)
        ax7.yaxis.tick_right()
        ax7.yaxis.set_label_position("right")

        norm = mpl.cm.colors.Normalize(vmax=1.0, vmin=0)
        ax8.imshow(rms2d_mean, cmap='jet', vmax=1.0, vmin=0, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax8.set_title('Both \nc) RMSE', fontsize=16)
        #ax0.set_ylabel('Alongshore (m)', fontsize=16)
        ax8.set_anchor('W')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax8)
        cbar.set_label('(m)', fontsize=12)

        norm = mpl.cm.colors.Normalize(vmax=.25, vmin=-.25)
        ax9.imshow(np.mean(diff_mean, axis=-1), cmap='bwr', vmin=-.25, vmax=.25, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax9.set_title('f) Bias', fontsize=16)
        ax9.set_xlabel('Cross-Shore (m)', fontsize=12)
        ax9.set_anchor('W')
        #ax1.set_ylabel('Alongshore (m)', fontsize=16)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=ax9)
        cbar.set_label('(m)', fontsize=12)

        ax10.hist(np.where(rms_histo < 1.5, rms_histo, .5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax10.set_title('j) RMSE Histogram', fontsize=16)
        ax10.set_xlabel('RMSE (m)', fontsize=12)
        ax10.set_ylabel('Count', fontsize=12)
        ax10.set_ylim(ymax=140)
        ax10.set_xlim(xmax=1.4)
        ax10.yaxis.tick_right()
        ax10.yaxis.set_label_position("right")

        ax11.hist(np.where(diff_histo > -1.5, diff_histo, -.5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax11.set_title('m) Bias Histogram', fontsize=16)
        ax11.set_xlabel('Bias (m)', fontsize=12)
        ax11.set_ylabel('Count', fontsize=12)
        ax11.set_ylim(ymax=200)
        ax11.set_xlim(xmax=1.5, xmin=-1.5)
        ax11.yaxis.tick_right()
        ax11.yaxis.set_label_position("right")
        plt.subplots_adjust(top=.62, hspace=.44, wspace=.46)

        # figure 3
        fig = plt.figure()
        grid = gridspec.GridSpec(6, 2, figure=fig)
        ax0 = fig.add_subplot(grid[:3, 0], projection='scatter_density')
        ax1 = fig.add_subplot(grid[:2, 1], projection='scatter_density')
        ax2 = fig.add_subplot(grid[3:, 0])
        #ax3 = fig.add_subplot(grid[0, 1], projection='scatter_density')
        ax4 = fig.add_subplot(grid[2:4, 1], projection='scatter_density')
        ax5 = fig.add_subplot(grid[4:, 1], projection='scatter_density')

        norm = ImageNormalize(img_mean, interval=ManualInterval(vmin=0, vmax=1), stretch=SqrtStretch())
        ax0.imshow(within_2d_mean, cmap='jet', norm=norm, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax0.set_title("a) Bounded by Ensemble", fontsize=16)
        ax0.set_ylabel("Alongshore (m)", fontsize=12)
        ax0.set_xlabel("Cross-shore (m)", fontsize=12)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=ImageNormalize(img_mean, interval=ManualInterval(vmin=0, vmax=100), stretch=SqrtStretch()), cmap='jet'), ax=ax0)
        cbar.set_label('% Bounded', fontsize=12)

        #create array of just bounded and nonbounded indices
        #plot histograms of labels, preds etc in 2 subplots bounded and unbounded
        bounded_indices = np.where(within1d == 1)
        not_bounded_indices = np.where(within1d == 0)
        ax1.hist(label1d[bounded_indices], bins=25, facecolor='green', edgecolor='black', linewidth=1.2,
                 label="Bounded", alpha=0.5)
        ax1.set_title('c) Pixel-wise Depths', fontsize=16)
        ax1.set_xlabel('Truth Depth (m)', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        ax1.hist(label1d[not_bounded_indices], bins=25, facecolor='blue', edgecolor='black', linewidth=1.2,
                 label="Unbounded", alpha=0.5)
        ax1.legend()
        imgnorm = ImageNormalize(img_mean, interval=ManualInterval(vmin=0, vmax=895), stretch=SqrtStretch())
        ax2.imshow(np.nansum(np.where(label_mean < -0.01, 1, 0), axis=-1), cmap='jet', norm=imgnorm, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax2.set_title("b) Underwater %", fontsize=16)
        ax2.set_ylabel("Alongshore (m)", fontsize=12)
        ax2.set_xlabel("Cross-shore (m)", fontsize=12)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=ImageNormalize(img_mean, interval=ManualInterval(vmin=0, vmax=100), stretch=SqrtStretch()), cmap='jet'), ax=ax2)
        cbar.set_label('% Subaqueous', fontsize=12)

        """ax3.hist(img1d[bounded_indices], bins=25, facecolor='green', edgecolor='black', linewidth=1.2,
                 label="Bounded", alpha=0.5)
        ax3.set_title('e) Pixel-wise Intensity', fontsize=16)
        ax3.set_xlabel('Grayscale Value', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.hist(img1d[not_bounded_indices], bins=25, facecolor='blue', edgecolor='black', linewidth=1.2,
                 label="Unbounded", alpha=0.5)
        ax3.legend()"""

        ax4.hist(mae1d[bounded_indices], bins=25, facecolor='green', edgecolor='black', linewidth=1.2,
                 label="Bounded", alpha=0.5)
        ax4.set_title('d) Pixel-wise Absolute Error', fontsize=16)
        ax4.set_xlabel('Absolute Error (m)', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        ax4.hist(mae1d[not_bounded_indices], bins=25, facecolor='blue', edgecolor='black', linewidth=1.2,
                 label="Unbounded", alpha=0.5)
        ax4.legend()
        ax5.hist(diff1d[bounded_indices], bins=25, facecolor='green', edgecolor='black', linewidth=1.2,
                 label="Bounded", alpha=0.5)
        ax5.set_title('e) Pixel-wise Bias', fontsize=16)
        ax5.set_xlabel('Bias (m)', fontsize=12)
        ax5.set_ylabel('Count', fontsize=12)
        ax5.yaxis.tick_right()
        ax5.yaxis.set_label_position("right")
        ax5.hist(diff1d[not_bounded_indices], bins=25, facecolor='blue', edgecolor='black', linewidth=1.2,
                 label="Unbounded", alpha=0.5)
        ax5.legend()
        plt.subplots_adjust(left=.14, bottom=.06, right=.67, top=.94, wspace=0.09, hspace=.98)

        # figure 4

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        to_plot = [H_rms3[0], H_rms2[0], H_rms[0], H_rms3[1], H_rms2[1], H_rms[1], H_rms3[2], H_rms2[2], H_rms[2], H_rms3[3], H_rms2[3], H_rms[3]]
        bp = ax1.boxplot(to_plot, positions=[1,2,3,5,6,7,9,10,11,13,14,15], patch_artist=True)
        ax1.set_xlabel('Height (m)', fontsize=12)
        ax1.set_ylabel('RMSE (m)', fontsize=12)
        bp, medians = make_boxplot(bp)
        ax1.set_ylim(ymax=1.5)
        nums = [i for i in H_bin]
        nums_avg = [np.mean(i) for i in H_bin]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([2, 6, 10, 14], labels)
        plt.plot([], c='pink', label='Timex')
        plt.plot([], c="lightblue", label='Snap')
        plt.plot([], c="lightgreen", label='Both')
        plt.legend()

        """ax2 = fig.add_subplot(1, 2, 2)
        to_plot = [H_bias3[0], H_bias2[0], H_bias[0], H_bias3[1], H_bias2[1], H_bias[1], H_bias3[2], H_bias2[2], H_bias[2], H_bias3[3], H_bias2[3], H_bias[3]]
        bp = ax2.boxplot(to_plot, positions=[1,2,3,5,6,7,9,10,11,13,14,15], patch_artist=True)
        ax2.set_xlabel('Height (m)', fontsize=16)
        ax2.set_ylabel('Bias (m)', fontsize=16)
        ax2.set_ylim(ymax=1, ymin=-1)
        plt.title("Wave Height", fontsize=20)
        bp, medians = make_boxplot(bp)
        nums = [i for i in H_bin]
        nums_avg = [np.mean(i) for i in H_bin]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([2, 6, 10, 14], labels)"""

        # Figure 6
        # find the -.01 farthest to the right for each row
        slopeindex = np.sum(np.any(label_mean > -.01, axis=0), axis=0)
        shoreline_elevation = np.mean(label_mean[:, slopeindex, :], axis=0)
        offshore_elevation = np.mean(label_mean[:, (100 - 1), :], axis=0)

        # divide by constant img_cols instead of (img_cols-slopeindex) to introduce noise into the slope "guess"
        shoreslope = np.mean((shoreline_elevation - offshore_elevation) / (100 - slopeindex), axis=0)
        irr = (shoreslope)/(np.sqrt(wc_list[:, 0]/(np.power((1/wc_list[:,2]), 2)*(9.81/(2*3.14)))))
        pshoreslope = np.gradient(label_mean)
        pirr = (pshoreslope) / (
            np.sqrt(wc_list[:, 0] / (np.power((1 / wc_list[:, 2]), 2) * (9.81 / (2 * 3.14)))))
        pshoreslope = pshoreslope[0].flatten()
        pirr = pirr[0].flatten()

        fig = plt.figure()
        grid = gridspec.GridSpec(2, 4, figure=fig)
        ax1 = fig.add_subplot(grid[0, 0], projection='scatter_density')
        ax2 = fig.add_subplot(grid[0, 1], projection='scatter_density')
        ax3 = fig.add_subplot(grid[0, 2], projection='scatter_density')
        ax4 = fig.add_subplot(grid[0, 3], projection='scatter_density')
        ax5 = fig.add_subplot(grid[1, 0], projection='scatter_density')
        ax6 = fig.add_subplot(grid[1, 1], projection='scatter_density')
        ax7 = fig.add_subplot(grid[1, 2], projection='scatter_density')
        ax8 = fig.add_subplot(grid[1, 3], projection='scatter_density')

        ax1.scatter(shoreslope, rms_histo)
        ax1.set_title("Image-wise")
        ax1.set_xlabel("Mean slope (m/m)")
        ax1.set_ylabel("RMSE (m)")
        ax2.scatter(shoreslope, diff_histo)
        ax2.set_title("Image-wise")
        ax2.set_xlabel("Mean slope (m/m)")
        ax2.set_ylabel("Bias (m)")
        ax3.scatter(irr, rms_histo)
        ax3.set_title("Image-wise")
        ax3.set_xlabel("Iribarren")
        ax3.set_ylabel("RMSE (m)")
        ax4.scatter(irr, diff_histo)
        ax4.set_title("Image-wise")
        ax4.set_xlabel("Iribarren")
        ax4.set_ylabel("Bias (m)")
        norm = ImageNormalize(vmin=0, vmax=1000, stretch=LogStretch())
        density = ax5.scatter_density(pshoreslope, mae1d, norm=norm, cmap='jet')
        ax5.set_title("Pixel-wise")
        ax5.set_xlabel("Slope (m/m)")
        ax5.set_ylabel("RMSE (m)")
        density = ax6.scatter_density(pshoreslope, diff1d, norm=norm, cmap='jet')
        ax6.set_title("Pixel-wise")
        ax6.set_xlabel("Slope (m/m)")
        ax6.set_ylabel("Bias (m)")
        density = ax7.scatter_density(pirr, mae1d, norm=norm, cmap='jet')
        ax7.set_title("Pixel-wise")
        ax7.set_xlabel("Iribarren")
        ax7.set_ylabel("AE (m)")
        density = ax8.scatter_density(pirr, diff1d, norm=norm, cmap='jet')
        ax8.set_title("Pixel-wise")
        ax8.set_xlabel("Iribarren")
        ax8.set_ylabel("Bias (m)")
        fig.colorbar(density, ax=ax8).set_label(label='Number of Points per Pixel', size=14)
        plt.subplots_adjust(hspace=.25, wspace=.25)

        plt.show()

    if args.indiv_pred_show:
        i = 0
        while i < test_set_length:
            print(i)
            #plot_for_gif2(img_mean, snap_mean, label_mean, pred_mean, diff_mean, i, pred_list, uncertainty_2d)
            plot_for_paper(img_mean, label_mean, pred_mean, diff_mean, pred_list,
                           img_mean2, pred_mean2, diff_mean2, pred_list2,
                           img_mean3, pred_mean3, diff_mean3, pred_list3, i)
            i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fs', '--fullstats', action='store_true',
                        help="print full stats and uncertainty of test set with N passes")
    parser.add_argument('-ips', '--indiv_pred_show', action='store_true', help="graphs indiv_preds")

    args = parser.parse_args()
    if real_or_fake == 'fake':
        img_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "img_mean.npy")
        snap_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "snap_mean.npy")
        label_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "label_mean.npy")
        pred_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "pred_mean.npy")
        diff_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "diff_mean.npy")
        rms2d_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "rms2d_mean.npy")
        mae2d_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "mae2d_mean.npy")
        uncertainty_2d = np.load('./results/plot/' + name1 + str(ensemble_runs) + "uncertainty_2d.npy")
        diff_histo = np.load('./results/plot/' + name1 + str(ensemble_runs) + "diff_histo.npy")
        rms_histo = np.load('./results/plot/' + name1 + str(ensemble_runs) + "rms_histo.npy")
        pred_list = np.load('./results/plot/' + name1 + str(ensemble_runs) + "pred_list.npy")
        within_2d = np.load('./results/plot/' + name1 + str(ensemble_runs) + "within_2d.npy")
        within_2d_mean = np.load('./results/plot/' + name1 + str(ensemble_runs) + "within_2d_mean.npy")
        img_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "img_mean.npy")
        snap_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "snap_mean.npy")
        label_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "label_mean.npy")
        pred_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "pred_mean.npy")
        diff_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "diff_mean.npy")
        rms2d_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "rms2d_mean.npy")
        mae2d_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "mae2d_mean.npy")
        uncertainty_2d2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "uncertainty_2d.npy")
        diff_histo2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "diff_histo.npy")
        rms_histo2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "rms_histo.npy")
        pred_list2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "pred_list.npy")
        within_2d2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "within_2d.npy")
        within_2d_mean2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "within_2d_mean.npy")
        img_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "img_mean.npy")
        snap_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "snap_mean.npy")
        label_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "label_mean.npy")
        pred_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "pred_mean.npy")
        diff_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "diff_mean.npy")
        rms2d_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "rms2d_mean.npy")
        mae2d_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "mae2d_mean.npy")
        uncertainty_2d3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "uncertainty_2d.npy")
        diff_histo3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "diff_histo.npy")
        rms_histo3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "rms_histo.npy")
        pred_list3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "pred_list.npy")
        within_2d3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "within_2d.npy")
        within_2d_mean3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "within_2d_mean.npy")
        mae1d = np.load('./results/plot/' + name1 + str(ensemble_runs) + "mae1d.npy")
        diff1d = np.load('./results/plot/' + name1 + str(ensemble_runs) + "diff1d.npy")
        within1d = np.load('./results/plot/' + name1 + str(ensemble_runs) + "within1d.npy")
        U_bin = np.load('./results/plot/' + name1 + str(ensemble_runs) + "U_bin.npy", allow_pickle=True)
        U_mae = np.load('./results/plot/' + name1 + str(ensemble_runs) + "U_mae.npy", allow_pickle=True)
        H_bin = np.load('./results/plot/' + name1 + str(ensemble_runs) + "H_bin.npy", allow_pickle=True)
        H_rms = np.load('./results/plot/' + name1 + str(ensemble_runs) + "H_rms.npy", allow_pickle=True)
        H_bias = np.load('./results/plot/' + name1 + str(ensemble_runs) + "H_bias.npy", allow_pickle=True)
        D_bin = np.load('./results/plot/' + name1 + str(ensemble_runs) + "D_bin.npy", allow_pickle=True)
        D_rms = np.load('./results/plot/' + name1 + str(ensemble_runs) + "D_rms.npy", allow_pickle=True)
        D_bias = np.load('./results/plot/' + name1 + str(ensemble_runs) + "D_bias.npy", allow_pickle=True)
        F_bin = np.load('./results/plot/' + name1 + str(ensemble_runs) + "F_bin.npy", allow_pickle=True)
        F_rms = np.load('./results/plot/' + name1 + str(ensemble_runs) + "F_rms.npy", allow_pickle=True)
        F_bias = np.load('./results/plot/' + name1 + str(ensemble_runs) + "F_bias.npy", allow_pickle=True)
        mae1d2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "mae1d.npy", allow_pickle=True)
        diff1d2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "diff1d.npy", allow_pickle=True)
        within1d2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "within1d.npy", allow_pickle=True)
        U_bin2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "U_bin.npy", allow_pickle=True)
        U_mae2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "U_mae.npy", allow_pickle=True)
        H_bin2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "H_bin.npy", allow_pickle=True)
        H_rms2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "H_rms.npy", allow_pickle=True)
        H_bias2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "H_bias.npy", allow_pickle=True)
        D_bin2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "D_bin.npy", allow_pickle=True)
        D_rms2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "D_rms.npy", allow_pickle=True)
        D_bias2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "D_bias.npy", allow_pickle=True)
        F_bin2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "F_bin.npy", allow_pickle=True)
        F_rms2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "F_rms.npy", allow_pickle=True)
        F_bias2 = np.load('./results/plot/' + name2 + str(ensemble_runs) + "F_bias.npy", allow_pickle=True)
        mae1d3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "mae1d.npy", allow_pickle=True)
        diff1d3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "diff1d.npy", allow_pickle=True)
        within1d3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "within1d.npy", allow_pickle=True)
        U_bin3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "U_bin.npy", allow_pickle=True)
        U_mae3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "U_mae.npy", allow_pickle=True)
        H_bin3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "H_bin.npy", allow_pickle=True)
        H_rms3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "H_rms.npy", allow_pickle=True)
        H_bias3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "H_bias.npy", allow_pickle=True)
        D_bin3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "D_bin.npy", allow_pickle=True)
        D_rms3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "D_rms.npy", allow_pickle=True)
        D_bias3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "D_bias.npy", allow_pickle=True)
        F_bin3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "F_bin.npy", allow_pickle=True)
        F_rms3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "F_rms.npy", allow_pickle=True)
        F_bias3 = np.load('./results/plot/' + name3 + str(ensemble_runs) + "F_bias.npy", allow_pickle=True)
        print("loaded")
        nrms = np.power(np.sum(np.power(np.where(label_mean < -.01, pred_mean - label_mean, 0), 2)) *
                        (1 / np.sum(np.power(np.where(label_mean < -.01, label_mean, 0), 2))), .5)
        model_mae = np.mean(mae2d_mean)
        model_bias = np.mean(diff_mean)
        model_rms = np.mean(rms2d_mean)
        model_nrms = np.mean(nrms)
        model_ninetyerror = np.percentile(np.abs(diff_mean), 90)
        nrms2 = np.power(np.sum(np.power(np.where(label_mean2 < -.01, pred_mean2 - label_mean2, 0), 2)) *
                        (1 / np.sum(np.power(np.where(label_mean2 < -.01, label_mean2, 0), 2))), .5)
        model_mae2 = np.mean(mae2d_mean2)
        model_bias2 = np.mean(diff_mean2)
        model_rms2 = np.mean(rms2d_mean2)
        model_nrms2 = np.mean(nrms2)
        model_ninetyerror2 = np.percentile(np.abs(diff_mean2), 90)
        nrms3 = np.power(np.sum(np.power(np.where(label_mean3 < -.01, pred_mean3 - label_mean3, 0), 2)) *
                        (1 / np.sum(np.power(np.where(label_mean3 < -.01, label_mean3, 0), 2))), .5)
        model_mae3 = np.mean(mae2d_mean3)
        model_bias3 = np.mean(diff_mean3)
        model_rms3 = np.mean(rms2d_mean3)
        model_nrms3 = np.mean(nrms3)
        model_ninetyerror3 = np.percentile(np.abs(diff_mean3), 90)
        print("both median ae: ", model_mae)
        print("both bias: ", model_bias)
        print("both rmse: ", model_rms)
        print("both nrmse: ", model_nrms)
        print("both 90 error: ", model_ninetyerror)
        print("both Within %: ", np.nanmean(within_2d))
        print("snap median ae: ", model_mae2)
        print("snap bias: ", model_bias2)
        print("snap rmse: ", model_rms2)
        print("snap nrmse: ", model_nrms2)
        print("snap 90 error: ", model_ninetyerror2)
        print("snap Within %: ", np.nanmean(within_2d2))
        print("timex median ae: ", model_mae3)
        print("timex bias: ", model_bias3)
        print("timex rmse: ", model_rms3)
        print("timex nrmse: ", model_nrms3)
        print("timex 90 error: ", model_ninetyerror3)
        print("timex Within %: ", np.nanmean(within_2d3))
    else:
        img_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "img_mean.npy")
        snap_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "snap_mean.npy")
        label_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "label_mean.npy")
        pred_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "pred_mean.npy")
        diff_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "diff_mean.npy")
        rms2d_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "rms2d_mean.npy")
        mae2d_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "mae2d_mean.npy")
        uncertainty_2d = np.load('./results/plot/' + name + str(ensemble_runs) + "uncertainty_2d.npy")
        diff_histo = np.load('./results/plot/' + name + str(ensemble_runs) + "diff_histo.npy")
        rms_histo = np.load('./results/plot/' + name + str(ensemble_runs) + "rms_histo.npy")
        pred_list = np.load('./results/plot/' + name + str(ensemble_runs) + "pred_list.npy")
        within_2d = np.load('./results/plot/' + name + str(ensemble_runs) + "within_2d.npy")
        """i = 0
        while i < test_set_length:
            print(i)
            plot_for_gif(img_mean, snap_mean, label_mean, pred_mean, diff_mean, i, pred_list, uncertainty_2d)
            i += 1
            plt.show()"""

    plot(img_mean, snap_mean, label_mean, pred_mean, diff_mean, rms2d_mean, mae2d_mean,
                 uncertainty_2d, diff_histo, rms_histo, pred_list, wc_list, within_2d, within_2d_mean,
         img_mean2, snap_mean2, label_mean2, pred_mean2, diff_mean2, rms2d_mean2, mae2d_mean2,
         uncertainty_2d2, diff_histo2, rms_histo2, pred_list2, within_2d2, within_2d_mean2,
         img_mean3, snap_mean3, label_mean3, pred_mean3, diff_mean3, rms2d_mean3, mae2d_mean3,
         uncertainty_2d3, diff_histo3, rms_histo3, pred_list3, within_2d3, within_2d_mean3,
         mae1d, diff1d, within1d, U_bin, U_mae, H_bin, H_rms, H_bias, D_bin, D_rms, D_bias, F_bin, F_rms, F_bias, mae1d2,
         diff1d2, within1d2, U_bin2, U_mae2, H_bin2, H_rms2, H_bias2, D_bin2, D_rms2, D_bias2, F_bin2, F_rms2, F_bias2, mae1d3,
         diff1d3, within1d3, U_bin3, U_mae3, H_bin3, H_rms3, H_bias3, D_bin3, D_rms3, D_bias3, F_bin3, F_rms3, F_bias3)
    imageio.mimsave('./' + name + str(noise_std) + '.gif',
                    [plot_for_gif2(img_mean, snap_mean, label_mean, pred_mean, diff_mean,
                                   i, pred_list, uncertainty_2d) for i in range(half_test_size)], fps=.5)
