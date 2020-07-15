# -*- coding:utf-8 -*-
from settings import *
from unet import TimexDataset, myUnet
from torch.utils.data.dataset import Dataset  # For custom data-sets
import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


class Predictor(object):
    def __init__(self):
        pass

    @staticmethod
    def main():
        test_dataset = TimexDataset(Dataset)
        test_unet = myUnet()

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
                square = 3
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
                which=0
                pred = predictions[-1]
                pred = pred[which, :, :, 0]

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
                                plt.imshow(img_batch[which, :, :img_cols, 0], cmap='gray')
                                plt.title("Timex")
                            #elif ix == 5:
                            #    plt.imshow(img_batch[0, :, :img_cols, 1], cmap='gray')
                            #    plt.title("Snap")
                            elif ix == 2:
                                plt.imshow(label_batch[which, :, :img_cols, 0], cmap='gist_earth', vmin=-6, vmax=1)
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
                            elif ix == 3:
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
                                    plt.imshow(fmap[which, :, :, ix - 1], cmap='gray')
                                    plt.title(model.layers[(layers_to_viz[l])].name, )
                                except:
                                    plt.imshow(batch1[0, :, :, ix - 1] - gauss1[0, :, :, ix-1], cmap='gray')
                                    plt.title("batch1 - gauss1")
                            ix += 1
                    l+=1
                    # show the figure
                    plt.subplots_adjust(right=.54)
                    plt.show()
                continue
            else:
                np.save('./results/' + name + '.npy', predictions)

            # calculate mean mae, mean bias, 10%tile error, 50%, 65%, 80%, 90%, greatest error,
            # also return the 2d means of pred, bias, and mae over the test set
            pred_cube, diff_cube, mae_cube, label_cube, timex_cube, snap_cube \
                = predict.calc_stats(img_batch, label_batch)

            # add each test set cube to a list over each ensemble run
            pred_list[:, :, :, i] = pred_cube
            diff_list[:, :, :, i] = diff_cube
            mae2d_list[:, :, :, i] = mae_cube
            label_list[:, :, :, i] = label_cube
            img_list[:, :, :, i] = timex_cube
            snap_list[:, :, :, i] = snap_cube

        # using infer-transformation use the flipped 2nd half prediction averaged with first half for more variation
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

        img_list[:, :, :, ensemble_runs:] = img_list[::-1, :, :, ensemble_runs:]
        snap_list[:, :, :, ensemble_runs:] = snap_list[::-1, :, :, ensemble_runs:]
        label_list[:, :, :, ensemble_runs:] = label_list[::-1, :, :, ensemble_runs:]
        pred_list[:, :, :, ensemble_runs:] = pred_list[::-1, :, :, ensemble_runs:]
        diff_list[:, :, :, ensemble_runs:] = diff_list[::-1, :, :, ensemble_runs:]
        mae2d_list[:, :, :, ensemble_runs:] = mae2d_list[::-1, :, :, ensemble_runs:]

        # calculate stats over the set of ensemble runs
        model_mae = np.mean(mae2d_list)
        model_bias = np.mean(diff_list)
        model_rms = np.power(np.sum(np.power(np.where(label_list < -.01, pred_list - label_list, 0), 2)) *
                        (1 / (test_size*ensemble_runs)), .5)
        model_nrms = np.power(np.sum(np.power(np.where(label_list < -.01, pred_list - label_list, 0), 2)) *
                        (1 / np.sum(np.power(np.where(label_list < -.01, label_list, 0), 2))), .5)
        model_ninetyerror = np.percentile(np.abs(diff_list), 90)
        print("Model median ae: ", model_mae)
        print("Model bias: ", model_bias)
        print("Model rmse: ", model_rms)
        print("Model nrmse: ", model_nrms)
        print("Model 90 error: ", model_ninetyerror)

        # average over each run to find ensemble means
        img_mean = np.mean(img_list, axis=3)
        snap_mean = np.mean(snap_list, axis=3)
        label_mean = np.mean(label_list, axis=3)
        pred_mean = np.mean(pred_list, axis=3)
        diff_mean = np.mean(diff_list, axis=3)
        mae2d_mean = np.mean(mae2d_list, axis=3)
        uncertainty_2d = 2*np.std(pred_list, axis=3)

        # for display of rms, difference of each pixel(x,y) over entire test set
        rms2d_mean = np.power(np.sum(np.power(diff_mean, 2), axis=-1) / test_set_length, .5)
        nrms = np.power(np.sum(np.power(np.where(label_mean < -.01, pred_mean - label_mean, 0), 2)) *
                        (1 / np.sum(np.power(np.where(label_mean < -.01, label_mean, 0), 2))), .5)
        diff_histo = np.mean(np.mean(diff_mean, axis=0), axis=0)
        rms_histo = np.power(np.sum(np.sum(np.power(diff_mean, 2), axis=0), axis=0)/(bathy_rows*(bathy_cols-downsample_zeroline)), .5)
        max_pred = np.amax(pred_list[:, downsample_zeroline:, :, :], axis=-1)
        min_pred = np.amin(pred_list[:, downsample_zeroline:, :, :], axis=-1)
        mean_mae = np.mean(mae2d_mean)
        mean_bias = np.mean(diff_mean)
        mean_rms = np.mean(rms2d_mean)
        mean_nrms = np.mean(nrms)
        mean_ninetyerror = np.percentile(diff_mean, 90)
        print("mean median ae: ", mean_mae)
        print("mean bias: ", mean_bias)
        print("mean rmse: ", mean_rms)
        print("mean nrmse: ", mean_nrms)
        print("mean 90 error: ", mean_ninetyerror)

        # calculate number of pixels whose prediction fall within ensemble range
        label_mean_within = np.where(label_mean > -.01, np.nan, label_mean)
        within_2d = np.where((max_pred > label_mean) & (label_mean > min_pred), 1, 0)
        within_2d = np.where(np.isnan(label_mean_within), label_mean_within, within_2d)
        within_2d_mean = np.nanmean(within_2d, axis=-1)
        print("within %: ", np.nanmean(within_2d))

        uc1d = uncertainty_2d.flatten()
        mae1d = mae2d_mean.flatten()
        diff1d = diff_mean.flatten()
        within1d = within_2d.flatten()

        U_bins = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 10]
        H_bins = [1.0, 1.4, 1.8, 3.0]
        D_bins = [65, 80, 95, 120]
        F_bins = [.1, .12, .14, .18]
        U_bin, U_mae, _ = create_bins(uc1d, U_bins, mae1d, mae1d)
        H_bin, H_rms, H_bias = create_bins(wc_list[:, 0], H_bins, rms_histo, diff_histo)
        D_bin, D_rms, D_bias = create_bins(wc_list[:, 1], D_bins, rms_histo, diff_histo)
        F_bin, F_rms, F_bias = create_bins(wc_list[:, 2], F_bins, rms_histo, diff_histo)

        np.save('./results/plot/' + name  + str(ensemble_runs)+ "img_mean.npy", img_mean)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "snap_mean.npy", snap_mean)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "label_mean.npy", label_mean)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "pred_mean.npy", pred_mean)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "diff_mean.npy", diff_mean)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "rms2d_mean.npy", rms2d_mean)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "mae2d_mean.npy", mae2d_mean)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "uncertainty_2d.npy", uncertainty_2d)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "diff_histo.npy", diff_histo)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "rms_histo.npy", rms_histo)
        np.save('./results/plot/' + name  + str(ensemble_runs)+ "pred_list.npy", pred_list)
        np.save('./results/plot/' + name + str(ensemble_runs) + "within_2d.npy", within_2d)
        np.save('./results/plot/' + name + str(ensemble_runs) + "within_2d_mean.npy", within_2d_mean)
        np.save('./results/plot/' + name + str(ensemble_runs) + "mae1d.npy", mae1d)
        np.save('./results/plot/' + name + str(ensemble_runs) + "diff1d.npy", diff1d)
        np.save('./results/plot/' + name + str(ensemble_runs) + "within1d.npy", within1d)
        np.save('./results/plot/' + name + str(ensemble_runs) + "U_bin.npy", U_bin)
        np.save('./results/plot/' + name + str(ensemble_runs) + "U_mae.npy", U_mae)
        np.save('./results/plot/' + name + str(ensemble_runs) + "H_bin.npy", H_bin)
        np.save('./results/plot/' + name + str(ensemble_runs) + "H_rms.npy", H_rms)
        np.save('./results/plot/' + name + str(ensemble_runs) + "H_bias.npy", H_bias)
        np.save('./results/plot/' + name + str(ensemble_runs) + "D_bin.npy", D_bin)
        np.save('./results/plot/' + name + str(ensemble_runs) + "D_rms.npy", D_rms)
        np.save('./results/plot/' + name + str(ensemble_runs) + "D_bias.npy", D_bias)
        np.save('./results/plot/' + name + str(ensemble_runs) + "F_bin.npy", F_bin)
        np.save('./results/plot/' + name + str(ensemble_runs) + "F_rms.npy", F_rms)
        np.save('./results/plot/' + name + str(ensemble_runs) + "F_bias.npy", F_bias)

    @staticmethod
    def calc_stats(img_batch, label_batch):

        # load data to write
        preds = np.load('./results/' + name + '.npy')

        # create lists for each statistic across the test set for this ensemble run

        # create cube for each object (timex, label, pred) and spatial derivatives for plotting
        timex_cube, snap_cube, label_cube, pred_cube = (np.zeros((bathy_rows, bathy_cols, np.size(preds,0))) for i in range(4))
        diff_cube, mae_cube = (np.zeros((bathy_rows, bathy_cols-downsample_zeroline, np.size(preds, 0))) for i in range(2))

        i=0
        # for each image in preds
        while i < len(preds):
            # grab first image, prediction image, and label
            img = img_batch[i]
            img = img[:, :, :-1]
            if snap and not snap_only:
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

            mae = np.power(np.power(np.where(label < -.01, pred-label, 0), 2), .5)
            difference = np.where(label < -.01, pred-label, 0)

            diff_cube[:, :, i] = difference
            mae_cube[:, :, i] = mae
            i += 1

        return pred_cube, diff_cube, mae_cube, label_cube, timex_cube, snap_cube


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-act', '--activations', action='store_true', help="check activations instead of predict")

    args = parser.parse_args()
    predict = Predictor()
    predict.main()
