# -*- coding:utf-8 -*-
from data import *


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
    #ax0 = fig.add_subplot(2, 3, 1), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
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

    img_norm = mpl.cm.colors.Normalize(vmax=.85, vmin=.4)
    img_cmap = mpl.cm.Greys_r

    ax0.imshow(img_mean[:, downsample_zeroline:, i]+.3, cmap='Greys_r',
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters], vmax=.85, vmin=.4)

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
    plt.tick_params(labelsize=14)

    ax1.imshow(label_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    plt.tick_params(labelsize=14)
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
    ax1.set_ylabel('Alongshore (m)', fontsize=14)
    ax1.set_xlabel('Cross-shore (m)', fontsize=14)


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

    ax3.imshow(snap_mean[:, downsample_zeroline:, i]+.3, cmap='Greys_r',
               extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters], vmax=.85, vmin=.4)
    # ax0 = fig.add_subplot(2, 3, 1), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
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
    ax3.set_ylabel('Alongshore (m)', fontsize=14)
    plt.tick_params(labelsize=14)

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
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=1, vmin=0), cmap=mpl.cm.inferno), ax=ax5)
    cbar.set_label('(m)', fontsize=14)

    fig.tight_layout(pad=3)
    #plt.subplots_adjust(right=.6)
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return image


def plot_for_paper(img_mean, label_mean, pred_mean, diff_mean, i, pred_list):

    X = np.linspace(zeroline, crosshore_distance_meters, crosshore_distance_meters-zeroline)
    Y = np.linspace(0, alongshore_distance_meters, alongshore_distance_meters)

    label_transect = np.mean(label_mean[:, :, i], axis=0)
    nrms_transect = np.power(np.sum(np.power(diff_mean[:, :, i], 2), axis=0)/np.sum(np.power(np.where(label_mean[:, :, i] != 0, label_mean[:, :, i], .01), 2), axis=0), .5)
    rms_transect = np.power(np.sum(np.power(diff_mean[:, :, i], 2), axis=0) / (bathy_cols), .5)

    label_up = cv2.resize(label_mean[:, :, i],
                          (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)
    pred_up = cv2.resize(pred_mean[:, :, i],
                         (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)

    cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m", "-3m", "-2.5m", "-2m",
                 "-1.5m", "-1m", "-.5m", "0m"]
    fmt = {}
    for l, s in zip([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], cs_labels):
        fmt[l] = s

    fig = plt.figure(figsize=(8, 6))
    grid = gridspec.GridSpec(2, 4, figure=fig)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[0, 3])
    ax4 = fig.add_subplot(grid[0, 3])
    ax5 = fig.add_subplot(grid[1, 0])


    norm = mpl.cm.colors.Normalize(vmax=1, vmin=-6)
    cmap = mpl.cm.gist_earth

    img_norm = mpl.cm.colors.Normalize(vmax=.9, vmin=.4)
    img_cmap = mpl.cm.Greys_r

    ax0.imshow(img_mean[:, downsample_zeroline:, i], cmap='Greys_r',
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters], vmax=1, vmin=0)
    #ax0 = fig.add_subplot(2, 3, 1), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
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
    ax0.set_title('a) Input', fontsize=16)
    ax0.set_ylabel('Alongshore (m)', fontsize=14)
    plt.tick_params(labelsize=14)


    ax1.imshow(pred_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    #ax2 = fig.add_subplot(2, 3, 3), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    ax1.set_title('b) Predicted', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax1.contour(X, Y, np.where(pred_up[:, :img_cols] > .1, 0, np.flip(pred_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax1.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax1.set_xlabel('Cross-shore (m)', fontsize=14)

    ax2.imshow(diff_mean[:, downsample_zeroline:, i], cmap='bwr', vmin=-2, vmax=2,
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
    ax2.set_xlabel('Cross-shore (m)', fontsize=14)
    ax2.set_title('c) Difference', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=.5, vmin=-.5), cmap=mpl.cm.bwr), ax=ax2)
    cbar.set_label('(m)', fontsize=14)

    pred_transects = [[] for j in range(2 * ensemble_runs)]
    for j in range(2 * ensemble_runs):
        pred_transects[j] = np.mean(pred_list[:, downsample_zeroline:, i, j], axis=0)
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
    plt.tick_params(labelsize=14)
    ax3.set_xlabel('Cross-shore (m)', fontsize=14)
    ax3.set_ylim(ymax=1, ymin=-8)

    x = np.linspace(zeroline, crosshore_distance_meters, num=len(rms_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    interpolate = interp.InterpolatedUnivariateSpline(x, rms_transect)
    ax4 = ax3.twinx()
    ax4.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='b', label='RMSE')
    interpolate = interp.InterpolatedUnivariateSpline(x, nrms_transect)
    ax4.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='black', label='Depth Norm RMSE')
    plt.title('d) Alongshore Average Transects', fontsize=16)
    ax4.legend(loc=1)
    ax3.legend(loc=6)
    ax4.set_ylabel('', fontsize=14)
    ax4.set_ylim(ymin=0, ymax=2)

    ax5.imshow(label_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    #ax1 = fig.add_subplot(2, 3, 2), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax5)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax5.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax5.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax5.set_title('i) Truth', fontsize=16)
    ax5.set_ylabel('Alongshore (m)', fontsize=14)
    plt.tick_params(labelsize=14)
    ax5.set_xlabel('Cross-shore (m)', fontsize=14)
    plt.subplots_adjust(top=.73, wspace=.40, hspace=.45)
    plt.show()
    # Used to return the plot as an image array
    #fig.canvas.draw()       # draw the canvas, cache the renderer
    #image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #plt.close('all')
    #return image


def make_boxplot(bp):
    for box in bp['boxes']:
        # change outline color
        box.set(color='black', linewidth=2)
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


def create_bins(arr, bins, errora, errorb):
    out = np.digitize(arr, bins, right=1)
    bin = [[] for i in range(len(bins))]
    a_value = [[] for i in range(len(bins))]
    b_value = [[] for i in range(len(bins))]
    for i in range(len(bins)):
        temp = np.where(out == i)
        bin[i] = (arr[temp])
        a_value[i]=(errora[temp])
        b_value[i]=(errorb[temp])

    return bin, a_value, b_value


def label_box(X_label, X_bin):
    nums = [X_label[0], X_label[1], X_label[2], X_label[3]]
    nums_avg = [np.mean(X_bin[0]), np.mean(X_bin[1]), np.mean(X_bin[2]), np.mean(X_bin[3])]
    labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
    plt.xticks([1, 2, 3, 4], labels)


def plot(img_mean, snap_mean, label_mean, pred_mean, diff_mean, rms2d_mean, mae2d_mean,
                 uncertainty_2d, diff_histo, rms_histo, pred_list, wc_list, within_2d, within_2d_mean):

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
        grid = gridspec.GridSpec(4, 1, figure=fig)
        ax0 = fig.add_subplot(grid[0, 0])
        ax1 = fig.add_subplot(grid[1, 0])
        ax2 = fig.add_subplot(grid[2, 0])
        ax3 = fig.add_subplot(grid[3, 0])

        norm = mpl.cm.colors.Normalize(vmax=.75, vmin=0)
        ax0.imshow(rms2d_mean, cmap='jet', vmax=.75, vmin=0, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax0.set_title('Both \nc) RMSE', fontsize=20)
        #ax0.set_ylabel('Alongshore (m)', fontsize=16)
        ax0.set_anchor('W')
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax0)
        cbar.set_label('(m)', fontsize=14)

        norm = mpl.cm.colors.Normalize(vmax=.25, vmin=-.25)
        ax1.imshow(np.mean(diff_mean, axis=-1), cmap='bwr', vmin=-.25, vmax=.25, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax1.set_title('f) Bias', fontsize=20)
        ax1.set_xlabel('Cross-Shore (m)', fontsize=16)
        ax1.set_anchor('W')
        #ax1.set_ylabel('Alongshore (m)', fontsize=16)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=ax1)
        cbar.set_label('(m)', fontsize=14)

        ax2.hist(np.where(rms_histo < 1.5, rms_histo, .5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax2.set_title('j) RMSE Histogram', fontsize=20)
        ax2.set_xlabel('RMSE (m)', fontsize=16)
        ax2.set_ylabel('Count', fontsize=16)
        ax2.set_ylim(ymax=130)
        ax2.set_xlim(xmax=1.4)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        ax3.hist(np.where(diff_histo > -1.5, diff_histo, -.5), bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
        ax3.set_title('m) Bias Histogram', fontsize=20)
        ax3.set_xlabel('Bias (m)', fontsize=16)
        ax3.set_ylabel('Count', fontsize=16)
        ax3.set_ylim(ymax=130)
        ax3.set_xlim(xmax=1.4)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        plt.subplots_adjust(right=.48, hspace=.36)

        # figure 3
        fig = plt.figure()
        grid = gridspec.GridSpec(2, 3, figure=fig)
        ax0 = fig.add_subplot(grid[0, 0], projection='scatter_density')
        ax1 = fig.add_subplot(grid[0, 1], projection='scatter_density')
        ax2 = fig.add_subplot(grid[1, 0])
        ax3 = fig.add_subplot(grid[0, 2], projection='scatter_density')
        ax4 = fig.add_subplot(grid[1, 1], projection='scatter_density')
        ax5 = fig.add_subplot(grid[1, 2], projection='scatter_density')

        norm = mpl.cm.colors.Normalize(vmax=100, vmin=0)
        ax0.imshow(within_2d_mean, cmap='jet', vmax=1, vmin=0, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax0.set_title("a) Bounded by Ensemble", fontsize=16)
        ax0.set_ylabel("Alongshore (m)", fontsize=12)
        ax0.set_xlabel("Cross-shore (m)", fontsize=12)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='jet'), ax=ax0)
        cbar.set_label('% Bounded', fontsize=14)

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
        ax2.imshow(np.nansum(np.where(label_mean < -0.01, 1, 0), axis=-1), cmap='jet', extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
        ax2.set_title("b) Inundation Rate", fontsize=16)
        ax2.set_ylabel("Alongshore (m)", fontsize=12)
        ax2.set_xlabel("Cross-shore (m)", fontsize=12)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=100, vmin=0), cmap='jet'), ax=ax2)
        cbar.set_label('% Subaqueous', fontsize=14)

        ax3.hist(uc1d[bounded_indices], bins=25, facecolor='green', edgecolor='black', linewidth=1.2,
                 label="Bounded", alpha=0.5)
        ax3.set_title('e) Pixel-wise Uncertainty', fontsize=16)
        ax3.set_xlabel('Uncertainty (m)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position("right")
        ax3.hist(uc1d[not_bounded_indices], bins=25, facecolor='blue', edgecolor='black', linewidth=1.2,
                 label="Unbounded", alpha=0.5)
        ax3.legend()
        ax4.hist(mae1d[bounded_indices], bins=25, facecolor='green', edgecolor='black', linewidth=1.2,
                 label="Bounded", alpha=0.5)
        ax4.set_title('d) Pixel-wise AE', fontsize=16)
        ax4.set_xlabel('Absolute Error (m)', fontsize=12)
        ax4.set_ylabel('Count', fontsize=12)
        ax4.yaxis.tick_right()
        ax4.yaxis.set_label_position("right")
        ax4.hist(mae1d[not_bounded_indices], bins=25, facecolor='blue', edgecolor='black', linewidth=1.2,
                 label="Unbounded", alpha=0.5)
        ax4.legend()
        ax5.hist(diff1d[bounded_indices], bins=25, facecolor='green', edgecolor='black', linewidth=1.2,
                 label="Bounded", alpha=0.5)
        ax5.set_title('f) Pixel-wise Bias', fontsize=16)
        ax5.set_xlabel('Bias (m)', fontsize=12)
        ax5.set_ylabel('Count', fontsize=12)
        ax5.yaxis.tick_right()
        ax5.yaxis.set_label_position("right")
        ax5.hist(diff1d[not_bounded_indices], bins=25, facecolor='blue', edgecolor='black', linewidth=1.2,
                 label="Unbounded", alpha=0.5)
        ax5.legend()
        plt.subplots_adjust(hspace=.25)

        # figure 4

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 3, 1)
        bp = ax1.boxplot([H_rms[0], H_rms[1], H_rms[2], H_rms[3]])
        ax1.set_xlabel('Height (m)', fontsize=16)
        ax1.set_ylabel('RMSE', fontsize=16)
        plt.title("Wave Height", fontsize=20)
        ax1.set_ylim(ymin=0, ymax=1)
        bp, medians = make_boxplot(bp)
        label_box(H_rms, H_bin)

        ax2 = fig.add_subplot(2, 3, 2)
        bp = ax2.boxplot([D_rms[0], D_rms[1], D_rms[2], D_rms[3]])
        bp, medians = make_boxplot(bp)
        ax2.set_xlabel('Direction (*)', fontsize=16)
        ax2.set_ylabel('RMSE', fontsize=16)
        ax2.set_ylim(ymin=0, ymax=1)
        label_box(D_rms, D_bin)
        plt.title("Wave Direction", fontsize=20)

        ax3 = fig.add_subplot(2, 3, 3)
        bp = ax3.boxplot([F_rms[0], F_rms[1], F_rms[2], F_rms[3]])
        bp, medians = make_boxplot(bp)
        plt.title("Wave Frequency", fontsize=20)
        ax3.set_xlabel('Frequency (Hz)', fontsize=16)
        ax3.set_ylabel('RMSE', fontsize=16)
        ax3.set_ylim(ymin=0, ymax=1)
        label_box(F_rms, F_bin)

        ax4 = fig.add_subplot(2, 3, 4)
        bp = ax4.boxplot([H_bias[0], H_bias[1], H_bias[2], H_bias[3]])
        label_box(H_bias, H_bin)
        ax4.set_xlabel('Height (m)', fontsize=16)
        ax4.set_ylabel('Bias', fontsize=16)
        plt.title("Wave Height", fontsize=20)
        ax4.set_ylim(ymin=-.75, ymax=.75)
        bp, medians = make_boxplot(bp)

        ax5 = fig.add_subplot(2, 3, 5)
        bp = ax5.boxplot([D_bias[0], D_bias[1], D_bias[2], D_bias[3]])
        bp, medians = make_boxplot(bp)
        label_box(D_bias, D_bin)
        ax5.set_xlabel('Direction (*)', fontsize=16)
        ax5.set_ylabel('Bias', fontsize=16)
        ax5.set_ylim(ymin=-.75, ymax=.75)
        plt.title("Wave Direction", fontsize=20)

        ax6 = fig.add_subplot(2, 3, 6)
        bp = ax6.boxplot([F_bias[0], F_bias[1], F_bias[2], F_bias[3]])
        bp, medians = make_boxplot(bp)
        label_box(F_bias, F_bin)
        ax6.set_ylim(ymin=-.75, ymax=.75)
        plt.title("Wave Frequency", fontsize=20)
        ax6.set_xlabel('Frequency (Hz)', fontsize=16)
        ax6.set_ylabel('Bias', fontsize=16)

        # Figure 6
        # find the -.01 farthest to the right for each row
        slopeindex = np.sum(np.any(label_mean > -.01, axis=0), axis=0)
        shoreline_elevation = np.mean(label_mean[:, slopeindex, :], axis=0)
        offshore_elevation = np.mean(label_mean[:, (100 - 1), :], axis=0)

        # divide by constant img_cols instead of (img_cols-slopeindex) to introduce noise into the slope "guess"
        shoreslope = np.mean((shoreline_elevation - offshore_elevation) / (100 - slopeindex), axis=0)
        irr = np.tan(shoreslope)/(np.sqrt(wc_list[:, 0]/(np.power((1/wc_list[:,2]), 2)*(9.81/(2*3.14)))))
        pshoreslope = np.gradient(label_mean)
        pirr = np.tan(pshoreslope) / (
            np.sqrt(wc_list[:, 0] / (np.power((1 / wc_list[:, 2]), 2) * (9.81 / (2 * 3.14)))))
        pshoreslope = pshoreslope[0].flatten()
        print(np.shape(pshoreslope))
        print(np.shape(pirr))
        pirr = pirr[0].flatten()
        print(np.shape(pirr))
        PS_bins = [.03, .05, .07, 3]
        S_bins = [.03, .05, .07, 3]
        I_bins = [.3, .5, .75, 3]
        PI_bins = [.3, .5, .75, 3]
        PS_out = np.digitize(pshoreslope, PS_bins, right=1)
        S_out = np.digitize(shoreslope, S_bins, right=1)
        I_out = np.digitize(irr, I_bins, right=1)
        PI_out = np.digitize(pirr, PI_bins, right=1)

        temp = np.where(PS_out==3)
        PS_bin4 = pshoreslope[temp]
        PS_rms4 = mae1d[temp]
        PS_bias4 = diff1d[temp]
        temp = np.where(PS_out==2)
        PS_bin3 = pshoreslope[temp]
        PS_rms3 = mae1d[temp]
        PS_bias3 = diff1d[temp]
        temp = np.where(PS_out==1)
        PS_bin2 = pshoreslope[temp]
        PS_rms2 = mae1d[temp]
        PS_bias2 = diff1d[temp]
        temp = np.where(PS_out==0)
        PS_bin1 = pshoreslope[temp]
        PS_rms1 = mae1d[temp]
        PS_bias1 = diff1d[temp]
        temp = np.where(PI_out==3)
        PI_bin4 = pirr[temp]
        PI_rms4 = mae1d[temp]
        PI_bias4 = diff1d[temp]
        temp = np.where(PI_out==2)
        PI_bin3 = pirr[temp]
        PI_rms3 = mae1d[temp]
        PI_bias3 = diff1d[temp]
        temp = np.where(PI_out==1)
        PI_bin2 = pirr[temp]
        PI_rms2 = mae1d[temp]
        PI_bias2 = diff1d[temp]
        temp = np.where(PI_out==0)
        PI_bin1 = pirr[temp]
        PI_rms1 = mae1d[temp]
        PI_bias1 = diff1d[temp]
        temp = np.where(S_out==3)
        S_bin4 = shoreslope[temp]
        S_rms4 = rms_histo[temp]
        S_bias4 = diff_histo[temp]
        temp = np.where(S_out==2)
        S_bin3 = shoreslope[temp]
        S_rms3 = rms_histo[temp]
        S_bias3 = diff_histo[temp]
        temp = np.where(S_out==1)
        S_bin2 = shoreslope[temp]
        S_rms2 = rms_histo[temp]
        S_bias2 = diff_histo[temp]
        temp = np.where(S_out==0)
        S_bin1 = shoreslope[temp]
        S_rms1 = rms_histo[temp]
        S_bias1 = diff_histo[temp]
        temp = np.where(I_out==3)
        I_bin4 = irr[temp]
        I_rms4 = rms_histo[temp]
        I_bias4 = diff_histo[temp]
        temp = np.where(I_out==2)
        I_bin3 = irr[temp]
        I_rms3 = rms_histo[temp]
        I_bias3 = diff_histo[temp]
        temp = np.where(I_out==1)
        I_bin2 = irr[temp]
        I_rms2 = rms_histo[temp]
        I_bias2 = diff_histo[temp]
        temp = np.where(I_out==0)
        I_bin1 = irr[temp]
        I_rms1 = rms_histo[temp]
        I_bias1 = diff_histo[temp]

        fig = plt.figure()
        ax1 = fig.add_subplot(2, 4, 1)
        bp = ax1.boxplot([S_rms1, S_rms2, S_rms3, S_rms4])
        ax1.set_xlabel('mean Slope (m/m)', fontsize=16)
        ax1.set_ylabel('RMSE', fontsize=16)
        plt.title("RMSE of Image Slope", fontsize=20)
        ax1.set_ylim(ymin=0, ymax=1.25)
        bp, medians = make_boxplot(bp)
        nums = [S_bin1, S_bin2, S_bin3, S_bin4]
        nums_avg = [np.mean(S_bin1), np.mean(S_bin2), np.mean(S_bin3), np.mean(S_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        ax2 = fig.add_subplot(2, 4, 2)
        bp = ax2.boxplot([I_rms1, I_rms2, I_rms3, I_rms4])
        ax2.set_xlabel('mean Iribarren #', fontsize=16)
        ax2.set_ylabel('RMSE', fontsize=16)
        plt.title("RMSE of Image Iribarren #", fontsize=20)
        ax2.set_ylim(ymin=0, ymax=1.25)
        bp, medians = make_boxplot(bp)
        nums = [I_bin1, I_bin2, I_bin3, I_bin4]
        nums_avg = [np.mean(I_bin1), np.mean(I_bin2), np.mean(I_bin3), np.mean(I_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        ax3 = fig.add_subplot(2, 4, 3)
        bp = ax3.boxplot([S_bias1, S_bias2, S_bias3, S_bias4])
        ax3.set_xlabel('mean Slope (m/m)', fontsize=16)
        ax3.set_ylabel('Bias', fontsize=16)
        plt.title("Bias of Image Slope", fontsize=20)
        ax3.set_ylim(ymin=-1, ymax=1)
        bp, medians = make_boxplot(bp)
        nums = [S_bin1, S_bin2, S_bin3, S_bin4]
        nums_avg = [np.mean(S_bin1), np.mean(S_bin2), np.mean(S_bin3), np.mean(S_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        ax4 = fig.add_subplot(2, 4, 4)
        bp = ax4.boxplot([I_bias1, I_bias2, I_bias3, I_bias4])
        ax4.set_xlabel('mean Iribarren #', fontsize=16)
        ax4.set_ylabel('Bias', fontsize=16)
        plt.title("Bias of Image Iribarren #", fontsize=20)
        ax4.set_ylim(ymin=-1, ymax=1)
        bp, medians = make_boxplot(bp)
        nums = [I_bin1, I_bin2, I_bin3, I_bin4]
        nums_avg = [np.mean(I_bin1), np.mean(I_bin2), np.mean(I_bin3), np.mean(I_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        ax5 = fig.add_subplot(2, 4, 5)
        bp = ax5.boxplot([PS_rms1, PS_rms2, PS_rms3, PS_rms4])
        ax5.set_xlabel('Slope (m/m)', fontsize=16)
        ax5.set_ylabel('AE', fontsize=16)
        plt.title("AE of Pixel Slope", fontsize=20)
        bp, medians = make_boxplot(bp)
        nums = [PS_bin1, PS_bin2, PS_bin3, PS_bin4]
        nums_avg = [np.mean(PS_bin1), np.mean(PS_bin2), np.mean(PS_bin3), np.mean(PS_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        ax6 = fig.add_subplot(2, 4, 6)
        bp = ax6.boxplot([PS_bias1, PS_bias2, PS_bias3, PS_bias4])
        ax6.set_xlabel('Slope (m/m)', fontsize=16)
        ax6.set_ylabel('Bias', fontsize=16)
        plt.title("Bias of Pixel Slope", fontsize=20)
        bp, medians = make_boxplot(bp)
        nums = [PS_bin1, PS_bin2, PS_bin3, PS_bin4]
        nums_avg = [np.mean(PS_bin1), np.mean(PS_bin2), np.mean(PS_bin3), np.mean(PS_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        ax7 = fig.add_subplot(2, 4, 7)
        bp = ax7.boxplot([PI_rms1, PI_rms2, PI_rms3, PI_rms4])
        ax7.set_xlabel('Iribarren #', fontsize=16)
        ax7.set_ylabel('AE', fontsize=16)
        plt.title("AE of Pixel Iribarren #", fontsize=20)
        bp, medians = make_boxplot(bp)
        nums = [PI_bin1, PI_bin2, PI_bin3, PI_bin4]
        nums_avg = [np.mean(PI_bin1), np.mean(PI_bin2), np.mean(PI_bin3), np.mean(PI_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        ax8 = fig.add_subplot(2, 4, 8)
        bp = ax8.boxplot([PI_bias1, PI_bias2, PI_bias3, PI_bias4])
        ax8.set_xlabel('Iribarren #', fontsize=16)
        ax8.set_ylabel('Bias', fontsize=16)
        plt.title("Bias of Pixel Iribarren #", fontsize=20)
        bp, medians = make_boxplot(bp)
        nums = [PI_bin1, PI_bin2, PI_bin3, PI_bin4]
        nums_avg = [np.mean(PI_bin1), np.mean(PI_bin2), np.mean(PI_bin3), np.mean(PI_bin4)]
        labels = ['{:.2f}\nn='.format(j) + str(np.size(nums[i])) for i, j in enumerate(nums_avg)]
        plt.xticks([1, 2, 3, 4], labels)
        plt.subplots_adjust(hspace=.35)
        plt.show()

    if args.indiv_pred_show:
        i = 0
        while i < test_set_length:
            #try:
            plot_for_gif2(img_mean, snap_mean, label_mean, pred_mean, diff_mean, i, pred_list, uncertainty_2d)
            i += 5
            plt.show()
            #except:
            #    continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fs', '--fullstats', action='store_true',
                        help="print full stats and uncertainty of test set with N passes")
    parser.add_argument('-ips', '--indiv_pred_show', action='store_true', help="graphs indiv_preds")

    args = parser.parse_args()

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
    within_2d_mean = np.load('./results/plot/' + name + str(ensemble_runs) + "within_2d_mean.npy")
    nrms = np.power(np.sum(np.power(np.where(label_mean < -.01, pred_mean - label_mean, 0), 2)) *
                    (1 / np.sum(np.power(np.where(label_mean < -.01, label_mean, 0), 2))), .5)
    model_mae = np.mean(mae2d_mean)
    model_bias = np.mean(diff_mean)
    mae_err = 2 * np.std(mae2d_mean)
    model_rms = np.mean(rms2d_mean)
    model_nrms = np.mean(nrms)
    rms_err = 2 * np.std(rms2d_mean)
    model_ninetyerror = np.percentile(diff_mean, 90)
    print("Model median ae: ", model_mae)
    print("Model bias: ", model_bias)
    print("Model mae uncertainty: ", mae_err)
    print("Model rmse: ", model_rms)
    print("Model nrmse: ", model_nrms)
    print("Model rmse uncertainty: ", rms_err)
    print("Model 90 error: ", model_ninetyerror)
    print("Within %: ", np.nanmean(within_2d))

    plot(img_mean, snap_mean, label_mean, pred_mean, diff_mean, rms2d_mean, mae2d_mean,
                 uncertainty_2d, diff_histo, rms_histo, pred_list, wc_list, within_2d, within_2d_mean)
    imageio.mimsave('./' + name + str(noise_std) + '.gif',
                    [plot_for_gif2(img_mean, snap_mean, label_mean, pred_mean, diff_mean,
                                   i, pred_list, uncertainty_2d) for i in range(half_test_size)], fps=.5)
