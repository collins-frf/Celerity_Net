# -*- coding:utf-8 -*-
from unet import *
import re

def plot_for_gif(img_mean, snap_mean, label_mean, pred_mean, diff_mean, i, pred_list):

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
    ax4 = fig.add_subplot(grid[0, 2])
    ax3 = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[1, 1])
    ax5 = fig.add_subplot(grid[1, 2])

    norm = mpl.cm.colors.Normalize(vmax=1, vmin=-6)
    cmap = mpl.cm.gist_earth

    img_norm = mpl.cm.colors.Normalize(vmax=.9, vmin=.1)
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
    ax0.set_title('a) Timex', fontsize=16)
    ax0.set_ylabel('Alongshore (m)', fontsize=14)
    plt.tick_params(labelsize=14)

    ax1.imshow(label_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    #ax1 = fig.add_subplot(2, 3, 2), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
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


    ax2.imshow(pred_mean[:, :, i], cmap='gist_earth', vmin=-4, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    #ax2 = fig.add_subplot(2, 3, 3), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    ax2.set_title('e) Predicted', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax2)
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax2.contour(X, Y, np.where(pred_up[:, :img_cols] > .1, 0, np.flip(pred_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax2.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    ax2.set_xlabel('Cross-shore (m)', fontsize=14)



    ax3.imshow(snap_mean[:, downsample_zeroline:, i], cmap='Greys_r',
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters], vmax=1, vmin=0)
    #ax3 = fig.add_subplot(2, 3, 1), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    cs = ax3.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-4, vmax=2, alpha=.5,
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
    ax3.set_title('d) Snap', fontsize=16)
    ax3.set_ylabel('Alongshore (m)', fontsize=14)
    ax3.set_xlabel('Cross-shore (m)', fontsize=14)
    plt.tick_params(labelsize=14)

    ax4.imshow(diff_mean[:, downsample_zeroline:, i], cmap='bwr', vmin=-2, vmax=2,
                                               extent=[zeroline, crosshore_distance_meters, 0,
                                                       alongshore_distance_meters])
    cs = ax4.contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)),
                        vmin=-4, vmax=2, alpha=.5,
                        colors='black',
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax4.clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    #ax4 = fig.add_subplot(2, 3, 2), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    ax4.set_xlabel('Cross-shore (m)', fontsize=14)
    ax4.set_title('c) Difference', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.cm.colors.Normalize(vmax=.5, vmin=-.5), cmap=mpl.cm.bwr), ax=ax4)
    cbar.set_label('(m)', fontsize=14)

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
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return image


class Predictor(object):
    def __init__(self):
        pass

    @staticmethod
    def main():
        test_dataset = TimexDataset(Dataset)
        test_unet = myUnet()

        mae_list, rms_list, nrms_list, greatest_list, ten_list, sixtyfive_list, fifty_list, eighty_list, ninety_list = \
            ([] for i in range(9))

        # create lists of 5m downsampled size bathy_rowsxbathy_cols of Test Set Size
        # and Number of Ensemble Runs to average over
        img_list, snap_list, label_list, pred_list = \
            (np.zeros((bathy_rows, bathy_cols, test_size, ensemble_runs)) for i in range(4))
        diff_list, mae2d_list = \
            (np.zeros((bathy_rows, bathy_cols-downsample_zeroline, test_size, ensemble_runs)) for i in range(2))
        #img_batch, label_batch = test_unet.get_batch(test_dataset, train_flag='test')
        for i in range(ensemble_runs):
            model = test_unet.load_model()

            if args.activations:
                layers_to_viz = [3, 4, 12, 21, 30, 47, 56, 65, 72]
                outputs = [model.layers[i + 1].output for i in layers_to_viz]
                model = tf.keras.Model(inputs=model.inputs, outputs=outputs)

            if i == 0:
                model.summary()
            tf.keras.backend.set_learning_phase(0)
            for layer in model.layers:
                layer.stddev = noise_std
            print("noise std: ", model.layers[5].stddev)

            img_batch, label_batch = test_unet.get_batch(test_dataset, train_flag='test')
            print("Ensemble Run #: ", i)

            # make a prediction on the entire test set
            predictions = model.predict(img_batch, batch_size=1, verbose=1)

            if args.activations:
            # plot the output from each block
                """square = 4
                X = np.linspace(0, img_cols, img_cols)
                Y = np.linspace(0, img_rows, img_rows)
                cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m",
                             "-3m", "-2.5m", "-2m", "-1.5m", "-1m", "-.5m", "0m"]
                fmt = {}
                for f, s in zip(
                        [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        cs_labels):
                    fmt[f] = s
                l=0
                batch1 = predictions[0]
                gauss1 = predictions[1]

                pred = predictions[-1]
                pred = pred[0, :, :, 0]

                for fmap in predictions:
                    # plot all 64 maps in an 8x8 squares
                    ix = 1
                    for _ in range(square):
                        for _ in range(square):
                            # specify subplot and turn of axis
                            ax = plt.subplot(square, square, ix)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # plot filter channel in grayscale
                            if ix == 1:
                                plt.imshow(img_batch[0, :, :img_cols, 0], cmap='gray')
                                plt.title("Timex")
                            elif ix == 5:
                                plt.imshow(img_batch[0, :, :img_cols, 1], cmap='gray')
                                plt.title("Snap")
                            elif ix == 9:
                                plt.imshow(label_batch[0, :, :img_cols, 0], cmap='gist_earth', vmin=-6, vmax=1)
                                cs = ax.contour(X, Y, np.where(label_batch[0, :, :img_cols, 0] > .1, 0, label_batch[0, :, :img_cols, 0]),
                                                    vmin=-6,
                                                    vmax=2, alpha=.5,
                                                    colors=['white', 'white', 'white', 'white', 'white', 'white',
                                                            'white',
                                                            'white', 'white', 'white', 'white', 'white', 'white',
                                                            'white',
                                                            'white', 'white', 'black'],
                                                    levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5,
                                                            -2,
                                                            -1.5, -1, -.5, -.01],
                                                    linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                                'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                                'solid', 'dashed', 'solid', 'dashed', 'solid'],
                                                    linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5,
                                                                1.5, .5,
                                                                1.5, .5, 2])
                                ax.clabel(cs,
                                              [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1,
                                               -.5,
                                               -.01], fmt=fmt, inline_spacing=2, fontsize='small', )
                                plt.title("Label")
                            elif ix == 13:
                                plt.imshow(pred[:, :img_cols], cmap='gist_earth', vmin=-6, vmax=1)
                                cs = ax.contour(X, Y, np.where(pred[:, :img_cols] > .1, 0, pred[:, :img_cols]),
                                                    vmin=-6,
                                                    vmax=2, alpha=.5,
                                                    colors=['white', 'white', 'white', 'white', 'white', 'white',
                                                            'white',
                                                            'white', 'white', 'white', 'white', 'white', 'white',
                                                            'white',
                                                            'white', 'white', 'black'],
                                                    levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5,
                                                            -2,
                                                            -1.5, -1, -.5, -.01],
                                                    linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                                'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                                'solid', 'dashed', 'solid', 'dashed', 'solid'],
                                                    linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5,
                                                                1.5, .5,
                                                                1.5, .5, 2])
                                ax.clabel(cs,
                                              [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1,
                                               -.5,
                                               -.01], fmt=fmt, inline_spacing=2, fontsize='small', )
                                plt.title("Pred")
                            else:
                                try:
                                    plt.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                                    plt.title(model.layers[(layers_to_viz[l])].name, )
                                except:
                                    plt.imshow(batch1[0, :, :, ix - 1] - gauss1[0, :, :, ix-1], cmap='gray')
                                    plt.title("batch1 - gauss1")
                            ix += 1
                    l+=1
                    # show the figure
                    plt.show()"""
                continue
            else:
                np.save('./results/' + name + '.npy', predictions)

            # calculate mean mae, mean bias, 10%tile error, 50%, 65%, 80%, 90%, greatest error,
            # also return the 2d means of pred, bias, and mae over the test set
            runmae, meanrms, meannrms, ten_percent, fifty_percent, sixtyfive_percent, eighty_percent, ninety_percent, \
            greatest_error, pred_cube, diff_cube, mae_cube, label_cube, timex_cube, snap_cube \
                = predict.calc_stats(img_batch, label_batch)

            # add each value to a list to expand over each ensemble run
            mae_list = np.append(mae_list, runmae)
            rms_list = np.append(rms_list, meanrms)
            nrms_list = np.append(nrms_list, meannrms)
            ninety_list = np.append(ninety_list, ninety_percent)

            # add each test set cube to a list over each ensemble run
            pred_list[:, :, :, i] = pred_cube
            diff_list[:, :, :, i] = diff_cube
            mae2d_list[:, :, :, i] = mae_cube
            label_list[:, :, :, i] = label_cube
            img_list[:, :, :, i] = timex_cube
            snap_list[:, :, :, i] = snap_cube

        #using infer-transformation use the flipped 2nd half prediction averaged with first half for more variation
        first_half = img_list[:, :, :half_test_size, :]
        second_half = img_list[:, :, half_test_size:, :]
        img_list = np.concatenate((first_half, second_half), axis=-1)
        first_half = snap_list[:, :, :half_test_size, :]
        second_half = snap_list[:, :, half_test_size:, :]
        snap_list = np.concatenate((first_half, second_half), axis=-1)
        first_half = label_list[:, :, :half_test_size, :]
        second_half = label_list[:, :, half_test_size:, :]
        label_list = np.concatenate((first_half, second_half), axis=-1)
        first_half = pred_list[:, :, :half_test_size, :]
        second_half = pred_list[:, :, half_test_size:, :]
        pred_list = np.concatenate((first_half, second_half), axis=-1)
        first_half = diff_list[:, :, :half_test_size, :]
        second_half = diff_list[:, :, half_test_size:, :]
        diff_list = np.concatenate((first_half, second_half), axis=-1)
        first_half = mae2d_list[:, :, :half_test_size, :]
        second_half = mae2d_list[:, :, half_test_size:, :]
        mae2d_list = np.concatenate((first_half, second_half), axis=-1)
        #img_list = (np.reshape(img_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #label_list = (np.reshape(label_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #pred_list = (np.reshape(pred_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #diff_list = (np.reshape(diff_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #mae2d_list = (np.reshape(mae2d_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #img_list = (np.reshape(img_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #label_list = (np.reshape(label_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #pred_list = (np.reshape(pred_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #diff_list = (np.reshape(diff_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        #mae2d_list = (np.reshape(mae2d_list, (bathy_rows, bathy_cols, half_test_size, ensemble_runs*2)))
        img_list[:, :, :, ensemble_runs:] = img_list[::-1, :, :, ensemble_runs:]
        snap_list[:, :, :, ensemble_runs:] = snap_list[::-1, :, :, ensemble_runs:]
        label_list[:, :, :, ensemble_runs:] = label_list[::-1, :, :, ensemble_runs:]
        pred_list[:, :, :, ensemble_runs:] = pred_list[::-1, :, :, ensemble_runs:]
        diff_list[:, :, :, ensemble_runs:] = diff_list[::-1, :, :, ensemble_runs:]
        mae2d_list[:, :, :, ensemble_runs:] = mae2d_list[::-1, :, :, ensemble_runs:]

        # calculate stats over the set of ensemble runs
        model_mae = np.mean(mae_list)
        mae_err = 2*np.std(mae_list)
        model_rms = np.mean(rms_list)
        model_nrms = np.mean(nrms_list)
        rms_err = 2*np.std(rms_list)
        model_ninetyerror = np.mean(ninety_list)
        print("Model median ae: ", model_mae)
        print("Model mae uncertainty: ", mae_err)
        print("Model rmse: ", model_rms)
        print("Model nrmse: ", model_nrms)
        print("Model rmse uncertainty: ", rms_err)
        print("Model 90 error: ", model_ninetyerror)

        # average over each run to find ensemble means
        img_mean = np.mean(img_list, axis=3)
        snap_mean = np.mean(snap_list, axis=3)
        label_mean = np.mean(label_list[:, downsample_zeroline:, :], axis=3)
        pred_mean = np.mean(pred_list[:, downsample_zeroline:, :], axis=3)
        diff_mean = np.mean(diff_list[:, :, :], axis=3)
        mae2d_mean = np.mean(mae2d_list[:, :, :], axis=3)
        pred_err = 2*np.std(pred_list[:, downsample_zeroline:, :], axis=3)

        # for display of rms, difference of each pixel(x,y) over entire test set
        rms2d_mean = np.power(np.sum(np.power(diff_mean, 2), axis=-1) / test_size, .5)
        rms2d_mean = np.power(np.sum(np.power(diff_mean, 2)/(-label_mean), axis=-1) / test_size, .5)
        diff_histo = np.mean(np.mean(diff_mean, axis=0), axis=0)
        rms_histo = np.power(np.sum(np.sum(np.power(diff_mean, 2), axis=0), axis=0)/(bathy_rows*(bathy_cols-downsample_zeroline)), .5)
        max_pred = np.amax(pred_list[:, downsample_zeroline:, :, :], axis=-1)
        min_pred = np.amin(pred_list[:, downsample_zeroline:, :, :], axis=-1)
        # calculate number of pixels whose prediction fall within ensemble range
        within_2d = np.where((max_pred > label_mean) & (label_mean > min_pred), 1, 0)
#        l = 0
#        while l < test_size:
#            plt.imshow(within_2d[:, :, l])
#            plt.show()
#            l+=10
        print("within %: ", np.sum(within_2d[:, 20:, :])/(bathy_rows*(bathy_cols-20)*half_test_size))

        # plot full stats, individual uncertainty, or individual predictions based on args
        predict.plot(img_mean, label_mean, pred_mean, diff_mean, rms2d_mean, mae2d_mean,
                     pred_err, diff_histo, rms_histo, pred_list)

        # save plots of individual predictions for each image in test set in a gif
        imageio.mimsave('./' + name + '.gif', [plot_for_gif(img_mean, snap_mean, label_mean, pred_mean, diff_mean,
                                                            i, pred_list) for i in range(half_test_size)], fps=.5)

    @staticmethod
    def calc_stats(img_batch, label_batch):

        # load data to write
        preds = np.load('./results/' + name + '.npy')

        # create lists for each statistic across the test set for this ensemble run
        runmae_list, runrms_list, runnrms_list, greatest_error_list, ten_percent_list, twentyfive_percent_list, \
            fifty_percent_list, eighty_percent_list, ninety_percent_list = ([] for i in range(9))

        # create cube for each object (timex, label, pred) and spatial derivatives for plotting
        timex_cube, snap_cube, label_cube, pred_cube = (np.zeros((bathy_rows, bathy_cols, np.size(preds,0))) for i in range(4))
        diff_cube, mae_cube = (np.zeros((bathy_rows, bathy_cols-downsample_zeroline, np.size(preds, 0))) for i in range(2))

        i=0
        # for each image in preds
        while i < len(preds):
            # grab first image, prediction image, and label
            img = img_batch[i]
            img = img[:, :, :-1]
            if snap:
                snap_img = img[:, :img_cols, 1]
                snap_img = cv2.resize(snap_img, (bathy_cols, bathy_rows), interpolation=cv2.INTER_AREA)
                snap_cube[:, :, i] = snap_img
            else:
                snap_img = np.zeros((bathy_rows, bathy_cols))
                snap_cube[:, :, i] = snap_img
            img = img[:, :img_cols, 0]
            #img = np.mean(img[:, :, 1], axis=2)

            # downsample to 5m grid resolution (resolution of measured bathys and comparable methods)

            img = cv2.resize(img, (bathy_cols, bathy_rows), interpolation=cv2.INTER_AREA)

            #img = np.expand_dims(img, axis=-1)
            pred = preds[i]
            pred = pred[:, :img_cols, 0]
            pred = cv2.resize(pred, (bathy_cols, bathy_rows), interpolation=cv2.INTER_AREA)
            pred[:, :downsample_zeroline] = 0

            label = label_batch[i]
            label = label[:, :img_cols,0]
            label = cv2.resize(label, (bathy_cols, bathy_rows), interpolation=cv2.INTER_AREA)
            label[:, :downsample_zeroline] = 0

            # experimental
            #pred = np.where(label == 0, 0, pred)
            pred = cv2.blur(pred, (10, 10))
            pred = np.where(pred > 0, 0, pred)
            """if i % 100 == 0:
                fig = plt.figure()
                X = np.linspace(0, bathy_cols, bathy_cols)
                Y = np.linspace(0, bathy_rows, bathy_rows)
                cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m", "-3m", "-2.5m",
                             "-2m", "-1.5m", "-1m", "-.5m", "0m"]
                fmt = {}
                for l, s in zip([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                                cs_labels):
                    fmt[l] = s
                ax = fig.add_subplot(1, 3, 1), plt.imshow(img[:, :])
                cs = ax[0].contour(X, Y, np.where(label[:, :] > .01, 0, label[:, :]), vmin=-6, vmax=2,
                                    alpha=.5,
                                    colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                            'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                                    levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5,
                                            -.01],
                                    linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                'solid'],
                                    linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
                ax[0].clabel(cs,
                              fmt=fmt, inline_spacing=2, fontsize='small', )
                plt.title("Timex")
                ax1 = fig.add_subplot(1, 3, 2), plt.imshow(label[:, :], cmap='gist_earth', vmin=-6, vmax=2)
                cs = ax1[0].contour(X, Y, np.where(label[:, :] > .1, 0, label[:, :]), vmin=-6, vmax=2,
                                    alpha=1,
                                    colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                            'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                                    levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5,
                                            -.01],
                                    linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                'solid'],
                                    linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
                ax1[0].clabel(cs,
                              fmt=fmt, inline_spacing=2, fontsize='small', )
                plt.title("Offshore Label")
                ax0 = fig.add_subplot(1, 3, 3), plt.imshow(pred[:, :], cmap='gist_earth', vmin=-6, vmax=2)
                cs = ax0[0].contour(X, Y, np.where(pred[:, :] > .1, 0, pred[:, :]), vmin=-6, vmax=2,
                                    alpha=.5,
                                    colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                            'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                                    levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5,
                                            -.01],
                                    linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                'solid'],
                                    linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
                ax0[0].clabel(cs,
                              fmt=fmt, inline_spacing=2, fontsize='small', )
                plt.title("Offshore Pred")

                plt.show()"""

            pred_cube[:, :, i] = pred
            label_cube[:, :, i] = label
            timex_cube[:, :, i] = img

            offshore_cutoff = np.sum(np.any(label > -.01, axis=0))
            mae = np.power(np.power((pred[:, downsample_zeroline:]-label[:, downsample_zeroline:]), 2), .5)
            nrms = np.power(np.sum(np.power(((pred[:, offshore_cutoff:]-label[:, offshore_cutoff:])/(-label[:, offshore_cutoff:])), 2))*
                           (1/(bathy_rows*(bathy_cols-offshore_cutoff))), .5)
            rms = np.power(np.sum(np.power(((pred[:, offshore_cutoff:] - label[:, offshore_cutoff:])), 2)) *
                           (1 / (bathy_rows * (bathy_cols - offshore_cutoff))), .5)
            difference = pred[:, downsample_zeroline:] - label[:, downsample_zeroline:]

            diff_cube[:, :, i] = difference
            mae_cube[:, :, i] = mae

            runmae = np.mean(mae[:, offshore_cutoff:])
            greatest_error = np.amax(np.absolute(difference[:, offshore_cutoff:]))
            ten_percent = np.percentile(np.absolute(difference[:, offshore_cutoff:]), 10)
            twentyfive_percent = np.percentile(np.absolute(difference[:, offshore_cutoff:]), 25)
            fifty_percent = np.percentile(np.absolute(difference[:, offshore_cutoff:]), 50)
            eighty_percent = np.percentile(np.absolute(difference[:, offshore_cutoff:]), 80)
            ninety_percent = np.percentile(np.absolute(difference[:, offshore_cutoff:]), 90)

            runmae_list = np.append(runmae_list, runmae)
            runrms_list = np.append(runrms_list, rms)
            runnrms_list = np.append(runnrms_list, nrms)
            greatest_error_list = np.append(greatest_error_list, greatest_error)
            ten_percent_list = np.append(ten_percent_list, ten_percent)
            twentyfive_percent_list = np.append(twentyfive_percent_list, twentyfive_percent)
            fifty_percent_list = np.append(fifty_percent_list, fifty_percent)
            eighty_percent_list = np.append(eighty_percent_list, eighty_percent)
            ninety_percent_list = np.append(ninety_percent_list, ninety_percent)

            i+=1

        model_mae = np.mean(runmae_list)
        model_rms = np.mean(runrms_list)
        model_nrms = np.mean(runnrms_list)
        model_greatesterror = np.mean(greatest_error_list)
        model_tenerror = np.mean(ten_percent_list)
        model_sixtyerror = np.mean(twentyfive_percent_list)
        model_fiftyerror = np.mean(fifty_percent_list)
        model_eightyerror = np.mean(eighty_percent_list)
        model_ninetyerror = np.mean(ninety_percent_list)

        return model_mae, model_rms, model_nrms, model_tenerror, model_fiftyerror, \
               model_sixtyerror, model_eightyerror, model_ninetyerror, model_greatesterror, \
               pred_cube, diff_cube, mae_cube, label_cube, timex_cube, snap_cube

    @staticmethod
    def plot(img_mean, label_mean, pred_mean, diff_mean, rms2d_mean, mae2d_mean, pred_err, diff_histo, rms_histo, pred_list):
        np.save('./results/plot/' + name + "img_mean.npy", img_mean)
        np.save('./results/plot/' + name + "label_mean.npy", label_mean)
        np.save('./results/plot/' + name + "pred_mean.npy", pred_mean)
        np.save('./results/plot/' + name + "diff_mean.npy", diff_mean)
        np.save('./results/plot/' + name + "rms2d_mean.npy", rms2d_mean)
        np.save('./results/plot/' + name + "mae2d_mean.npy", mae2d_mean)
        np.save('./results/plot/' + name + "pred_err.npy", pred_err)
        np.save('./results/plot/' + name + "diff_histo.npy", diff_histo)
        np.save('./results/plot/' + name + "rms_histo.npy", rms_histo)
        np.save('./results/plot/' + name + "pred_list.npy", pred_list)
        if args.fullstats:
            mpl.rcParams['agg.path.chunksize'] = zeroline


            img1d = img_mean[:, downsample_zeroline:].flatten()
            pred1d = pred_mean.flatten()
            label1d = label_mean.flatten()
            uc1d = pred_err.flatten()
            mae1d = mae2d_mean.flatten()
            diff1d = diff_mean.flatten()

            bins = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 10]
            out = np.digitize(uc1d, bins, right=1)
            print(np.shape(out))
            print(out)

            i=0
            temp = np.where(out==9)
            mae1d_9 = mae1d[temp]
            temp = np.where(out==8)
            mae1d_8 = mae1d[temp]
            temp = np.where(out==7)
            mae1d_7 = mae1d[temp]
            temp = np.where(out==6)
            mae1d_6 = mae1d[temp]
            temp = np.where(out==5)
            mae1d_5 = mae1d[temp]
            temp = np.where(out==4)
            mae1d_4 = mae1d[temp]
            temp = np.where(out==3)
            mae1d_3 = mae1d[temp]
            temp = np.where(out==2)
            mae1d_2 = mae1d[temp]
            temp = np.where(out==1)
            mae1d_1 = mae1d[temp]
            temp = np.where(out==0)
            mae1d_0 = mae1d[temp]

            #z = np.polyfit(pred1d, mae1d, 1)
            #p = np.poly1d(z)
            #fit = p(pred1d)

            #intense_z = np.polyfit(img1d, mae1d, 1)
            #intense_p = np.poly1d(z)
            #intense_fit = intense_p(img1d)

            # figure 1
            fig = plt.figure()
            grid = gridspec.GridSpec(2, 3, figure=fig)
            ax0 = fig.add_subplot(grid[:, 0], projection='scatter_density')
            ax1 = fig.add_subplot(grid[0, 1])
            ax3 = fig.add_subplot(grid[0, 2])
            ax2 = fig.add_subplot(grid[1, 1])
            ax4 = fig.add_subplot(grid[1, 2])

            norm = ImageNormalize(vmin=0, vmax=1000, stretch=LogStretch())
            cmap = mpl.cm.jet
            density = ax0.scatter_density(-pred1d, -label1d, norm=norm, cmap='jet')
            fig.colorbar(density, ax=ax0).set_label(label='Number of Points per Pixel', size=14)
            ax0.plot([0, 8], [0, 8], c='black')
            ax0.set_ylim(0, 8)
            #ax0.text(5, 6, '1:1', fontsize=14, color='black')
            ax0.set_title('a) Pred vs Truth', fontsize=20)
            ax0.set_xlabel('Prediction Depth (m)', fontsize=16)
            ax0.set_ylabel('Truth Depth (m)', fontsize=16)

            norm = mpl.cm.colors.Normalize(vmax=.2, vmin=0)
            ax1.imshow(rms2d_mean, cmap='jet', vmax=.3, vmin=0, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
            ax1.set_title('b) Depth Norm RMSE', fontsize=20)
            ax1.set_anchor('W')
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
            cbar.set_label('(m)', fontsize=14)

            norm = mpl.cm.colors.Normalize(vmax=-np.mean(diff_mean, axis=-1).min(), vmin=np.mean(diff_mean, axis=-1).min())
            ax2.imshow(np.mean(diff_mean, axis=-1), cmap='bwr', vmin=np.mean(diff_mean, axis=-1).min(), vmax=-np.mean(diff_mean, axis=-1).min(), extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
            ax2.set_title('c) Bias', fontsize=20)
            ax2.set_xlabel('Cross-Shore (m)', fontsize=16)
            ax2.set_anchor('W')
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='bwr'), ax=ax2)
            cbar.set_label('(m)', fontsize=14)

            ax3.hist(rms_histo, bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
            ax3.set_title('d) RMSE Histogram', fontsize=20)
            ax3.set_xlabel('RMSE (m)', fontsize=16)
            ax3.set_ylabel('Count', fontsize=16)
            ax3.yaxis.tick_right()
            ax3.yaxis.set_label_position("right")

            ax4.hist(diff_histo, bins=25, facecolor='green', edgecolor='black', linewidth=1.2)
            ax4.set_title('e) Bias Histogram', fontsize=20)
            ax4.set_xlabel('Bias (m)', fontsize=16)
            ax4.set_ylabel('Count', fontsize=16)
            ax4.yaxis.tick_right()
            ax4.yaxis.set_label_position("right")

            # figure 2
            fig = plt.figure()

            ax1 = fig.add_subplot(1, 2, 1)
            bp = ax1.boxplot([mae1d_0, mae1d_1, mae1d_2, mae1d_3, mae1d_4,
                         mae1d_5, mae1d_6, mae1d_7, mae1d_8, mae1d_9])
            plt.title('Uncertainty Map', fontsize=20)
            plt.xlabel('Uncertainty Value Bin', fontsize=16)
            plt.ylabel('Ensemble MAE', fontsize=16)
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

            #pred_mean = np.absolute(np.mean(pred_list[:, downsample_zeroline:, :], axis=3))
            plt.show()

        if args.indiv_pred_show:
            i = 0
            while i < test_size:
                plot_for_gif(img_mean, label_mean, pred_mean, diff_mean, i, pred_list)
                plt.show()
                i+=10

        if args.indiv_uncertain:
            j=0
            uncertainty = np.where((pred_err[:, :, :] > .9), medians[9], pred_err)
            uncertainty = np.where((uncertainty[:, :, :] > .8) & (uncertainty[:, :, :] <.9), medians[8], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .7) & (uncertainty[:, :, :] < .8), medians[7], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .6) & (uncertainty[:, :, :] < .7), medians[6], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .5) & (uncertainty[:, :, :] < .6), medians[5], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .4) & (uncertainty[:, :, :] < .5), medians[4], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .3) & (uncertainty[:, :, :] < .4), medians[3], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .2) & (uncertainty[:, :, :] < .3), medians[2], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .1) & (uncertainty[:, :, :] < .2), medians[1], uncertainty)
            uncertainty = np.where((uncertainty[:, :, :] > .0) & (uncertainty[:, :, :] < .1), medians[0], uncertainty)

            while j < test_size:
                print(np.shape(pred_mean[:, downsample_zeroline:, j]))
                X = np.linspace(zeroline, crosshore_distance_meters, crosshore_distance_meters-zeroline)
                Y = np.linspace(0, alongshore_distance_meters, alongshore_distance_meters)
                fig = plt.figure()
                fig.add_subplot(2, 2, 1), plt.imshow(img_mean[:, downsample_zeroline:, j], cmap='Greys_r', extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
                plt.title('a) Timex', fontsize=10)
                plt.tick_params(labelsize=10)
                plt.ylabel('Alongshore (m)', fontsize=10)
                fig.add_subplot(2, 2, 2), plt.imshow(-pred_mean[:, downsample_zeroline:, j], cmap='gist_earth', vmin=-6, vmax=4, extent=[zeroline,crosshore_distance_meters,0,alongshore_distance_meters])
                cbar = plt.colorbar()
                cbar.set_label('Elevation (m)', fontsize=10)
                plt.contour(X, Y, np.flip(-cv2.resize(pred_mean[:, downsample_zeroline:, j], (crosshore_distance_meters- zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC),axis=0), colors='white', vmin=-6, vmax=4)
                plt.title('b) Ensemble Prediction', fontsize=10)
                plt.tick_params(labelsize=10)
                fig.add_subplot(2, 2, 3), plt.imshow(mae2d_mean[:, :, j],cmap='jet', vmin=0, vmax=1, extent=[zeroline, crosshore_distance_meters, 0,alongshore_distance_meters])
                plt.title('c) Absolute Error', fontsize=10)
                plt.ylabel('Alongshore (m)', fontsize=10)
                plt.xlabel('Cross-shore (m)', fontsize=10)
                cbar = plt.colorbar()
                cbar.set_label('Error (m)', fontsize=10)
                plt.tick_params(labelsize=10)
                fig.add_subplot(2, 2, 4), plt.imshow(uncertainty[:, :, j], cmap='jet', vmin=0, vmax=1, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
                plt.title('d) Uncertainty', fontsize=10)
                plt.xlabel('Cross-shore (m)', fontsize=10)
                cbar = plt.colorbar()
                cbar.set_label('Projected Error', fontsize=10)
                plt.tick_params(labelsize=10)
                plt.show()
                j+=10


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-fs', '--fullstats', action='store_true',
                        help="print full stats and uncertainty of test set with N passes")
    parser.add_argument('-iu', '--indiv_uncertain', action='store_true',
                        help="print out examples of individual uncertainty")
    parser.add_argument('-ips', '--indiv_pred_show', action='store_true', help="graphs indiv_preds")
    parser.add_argument('-act', '--activations', action='store_true', help="check activations instead of predict")

    args = parser.parse_args()
    predict = Predictor()
    predict.main()
