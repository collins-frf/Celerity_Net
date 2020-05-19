# -*- coding:utf-8 -*-
from unet import *

def plot_for_gif(imgmean, labelmean, predmean, diffmean, i, pred_list):

    X = np.linspace(zeroline, crosshore_distance_meters, crosshore_distance_meters-zeroline)
    Y = np.linspace(0, alongshore_distance_meters, alongshore_distance_meters)

    label_transect = np.mean(labelmean[:, :, i], axis=0)
    rms_transect = np.power(np.sum(np.power(diffmean[:, :, i], 2), axis=0) / (bathy_cols-downsample_zeroline), .5)

    label_up = cv2.resize(labelmean[:, :, i],
                          (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)
    pred_up = cv2.resize(predmean[:, :, i],
                         (crosshore_distance_meters-zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC)

    cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m", "-3m", "-2.5m", "-2m",
                 "-1.5m", "-1m", "-.5m", "0m"]
    fmt = {}
    for l, s in zip([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], cs_labels):
        fmt[l] = s

    fig = plt.figure()
    fig.set_size_inches(16, 9)

    ax0 = fig.add_subplot(2, 3, 1), \
          plt.imshow(imgmean[:, downsample_zeroline:, i], cmap='Greys_r',
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters], vmax=1, vmin=0)
    #ax0 = fig.add_subplot(2, 3, 1), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    cs = ax0[0].contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-6, vmax=2, alpha=.5,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    print(cs.levels)
    ax0[0].clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    cbar = plt.colorbar()
    cbar.set_label('Pixel Intensity', fontsize=14)
    plt.title('a) Timex', fontsize=16)
    plt.ylabel('Alongshore (m)', fontsize=14)
    plt.tick_params(labelsize=14)

    ax1 = fig.add_subplot(2, 3, 2), \
          plt.imshow(labelmean[:, :, i], cmap='gist_earth', vmin=-6, vmax=1,
                     extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    #ax1 = fig.add_subplot(2, 3, 2), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    cbar = plt.colorbar()
    cbar.set_label('Elevation (m)', fontsize=14)
    plt.tick_params(labelsize=14)
    cs = ax1[0].contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-6, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax1[0].clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    plt.title('b) Truth', fontsize=16)

    pred_transects = [[] for j in range(uncertainty_runs)]
    for j in range(uncertainty_runs):
        pred_transects[j] = np.mean(pred_list[:, downsample_zeroline:, i, j],axis=0)
    x = np.linspace(zeroline, crosshore_distance_meters, num=len(label_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    label_interpolate = interp.InterpolatedUnivariateSpline(x, label_transect)

    for j in range(len(pred_transects) - 1):
        pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[j])
        ax2 = fig.add_subplot(2, 3, 3), \
              plt.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey')

    pred_interpolate = interp.InterpolatedUnivariateSpline(x, pred_transects[-1])
    ax2 = fig.add_subplot(2, 3, 3), \
          plt.plot(np.linspace(zeroline, crosshore_distance_meters), pred_interpolate(x_new), c='grey', label='Individual Predictions')
    pred_interpolate = interp.InterpolatedUnivariateSpline(x, np.mean(pred_transects, axis=0))
    ax2 = fig.add_subplot(2, 3, 3), \
          plt.plot(x_new, pred_interpolate(x_new), c='red', label='Mean Prediction')
    ax2 = fig.add_subplot(2, 3, 3), \
          plt.plot(np.linspace(zeroline, crosshore_distance_meters), label_interpolate(x_new), c='cyan', label='Truth')
    ax2[0].yaxis.tick_right()
    ax2[0].yaxis.set_label_position("right")
    plt.tick_params(labelsize=14)
    plt.title('c) Alongshore Average Transects', fontsize=16)
    plt.ylabel('Elevation (m)', fontsize=14)
    plt.legend()

    ax3 = fig.add_subplot(2, 3, 4), plt.imshow(diffmean[:, downsample_zeroline:, i], cmap='bwr', vmin=-2, vmax=2,
                                               extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    cs = ax3[0].contour(X, Y, np.where(label_up[:, :img_cols] > .1, 0, np.flip(label_up[:, :img_cols], axis=0)), vmin=-6, vmax=2, alpha=.5,
                        colors='black',
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax3[0].clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )
    #ax3 = fig.add_subplot(2, 3, 4), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    plt.ylabel('Alongshore (m)', fontsize=14)
    plt.xlabel('Cross-shore (m)', fontsize=14)
    plt.title('d) Difference', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = plt.colorbar()
    cbar.set_label('(m)', fontsize=14)

    ax4 = fig.add_subplot(2, 3, 5), plt.imshow(predmean[:, :, i], cmap='gist_earth', vmin=-6, vmax=1,
                                               extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
    #ax4 = fig.add_subplot(2, 3, 5), plt.plot([zeroline, crosshore_distance_meters], [258, 258], color='r')
    plt.xlabel('Cross-shore (m)', fontsize=14)
    plt.title('e) Predicted', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = plt.colorbar()
    cbar.set_label('Elevation (m)', fontsize=14)
    cs = ax4[0].contour(X, Y, np.where(pred_up[:, :img_cols] > .1, 0, np.flip(pred_up[:, :img_cols], axis=0)), vmin=-6, vmax=2, alpha=1,
                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid',
                                    'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
    ax4[0].clabel(cs, fmt=fmt,
                  inline_spacing=2, fontsize='x-small', )

    x = np.linspace(zeroline, crosshore_distance_meters, num=len(rms_transect))
    x_new = np.linspace(zeroline, crosshore_distance_meters)
    interpolate = interp.InterpolatedUnivariateSpline(x, rms_transect)
    ax5 = fig.add_subplot(2, 3, 6), plt.plot(np.linspace(zeroline, crosshore_distance_meters), interpolate(x_new), c='r')
    ax5[0].yaxis.tick_right()
    ax5[0].yaxis.set_label_position("right")
    plt.title('f) Alongshore Averaged RMSE', fontsize=16)
    #plt.ylim((0, 1))
    plt.xlabel('Cross-shore (m)', fontsize=14)
    plt.ylabel('Error (m)', fontsize=14)
    plt.tick_params(labelsize=14)
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

    def eval(self):
        test_dataset = TimexDataset(Dataset)
        unet = myUnet()
        model = unet.load_model()

        mae_list = []
        rms_list = []
        greatest_list = []
        ten_list = []
        sixtyfive_list = []
        fifty_list = []
        eighty_list = []
        ninety_list = []
        #create lists of size HxW by Test Set Size by Number of Ensemble Runs to average over
        img_list = np.zeros((bathy_rows, bathy_cols, test_size, uncertainty_runs))
        label_list = np.zeros((bathy_rows, bathy_cols, test_size, uncertainty_runs))
        pred_list = np.zeros((bathy_rows, bathy_cols, test_size, uncertainty_runs))
        diff_list = np.zeros((bathy_rows, bathy_cols-downsample_zeroline, test_size, uncertainty_runs))
        mae2d_list = np.zeros((bathy_rows, bathy_cols-downsample_zeroline, test_size, uncertainty_runs))

        for i in range(uncertainty_runs):
            print(i)
            imgdatas, imglabels = unet.get_batch(test_dataset, train_flag='test')
            """for j in range(len(imgdatas)):
                if j % 100 == 0:
                    image = imgdatas[j]
                    label = imglabels[j]
                    print(j)
                    print(np.shape(image))
                    print(np.shape(label))
                    fig = plt.figure()
                    X = np.linspace(0, img_cols, img_cols)
                    Y = np.linspace(0, img_rows, img_rows)
                    cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m",
                                 "-3m", "-2.5m", "-2m", "-1.5m", "-1m", "-.5m", "0m"]
                    fmt = {}
                    for f, s in zip(
                            [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                            cs_labels):
                        fmt[f] = s
                    ax0 = fig.add_subplot(1, 2, 1), plt.imshow(image[:, :img_cols, 0])
                    cs = ax0[0].contour(X, Y, np.where(label[:, :img_cols, 0] > .1, 0, label[:, :img_cols, 0]), vmin=-6,
                                        vmax=2, alpha=.5,
                                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white',
                                                'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                                'white', 'white', 'black'],
                                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2,
                                                -1.5, -1, -.5, -.01],
                                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                    'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                    'solid', 'dashed', 'solid', 'dashed', 'solid'],
                                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5,
                                                    1.5, .5, 2])
                    ax0[0].clabel(cs,
                                  [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5,
                                   -.01], fmt=fmt, inline_spacing=2, fontsize='small', )
                    ax1 = fig.add_subplot(1, 2, 2), plt.imshow(label[:, :img_cols, 0], cmap='gist_earth', vmin=-6,
                                                               vmax=1)
                    cs = ax1[0].contour(X, Y, np.where(label[:, :img_cols, 0] > .1, 0, label[:, :img_cols, 0]), vmin=-6,
                                        vmax=2, alpha=1,
                                        colors=['white', 'white', 'white', 'white', 'white', 'white', 'white',
                                                'white', 'white', 'white', 'white', 'white', 'white', 'white',
                                                'white', 'white', 'black'],
                                        levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2,
                                                -1.5, -1, -.5, -.01],
                                        linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                    'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed',
                                                    'solid', 'dashed', 'solid', 'dashed', 'solid'],
                                        linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5,
                                                    1.5, .5, 2])
                    ax1[0].clabel(cs,
                                  [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5,
                                   -.01], fmt=fmt, inline_spacing=2, fontsize='small', )
                    plt.show()"""

            imgs_mask_test = model.predict(imgdatas, batch_size=1, verbose=1)
            np.save('./results/mask_test.npy', imgs_mask_test)

            #make a prediction on the entire test set and return
            #mean mae, mean bias, 10%tile error, 50%, 65%, 80%, 90%, greatest error,
            #also return the 2d versions of pred, bias, and mae for the test set
            meanmae, meanrms, ten_percent, fifty_percent, \
            sixtyfive_percent, eighty_percent, ninety_percent, greatest_error, pred_cube,\
                diff_cube, mae_cube, label_cube, timex_cube = predict.calc_stats(imgdatas, imglabels)

            #add each value to a list to expand over each uncertainty run
            mae_list = np.append(mae_list, meanmae)
            rms_list = np.append(rms_list, meanrms)
            greatest_list = np.append(greatest_list, greatest_error)
            ten_list = np.append(ten_list, ten_percent)
            sixtyfive_list = np.append(sixtyfive_list, sixtyfive_percent)
            fifty_list = np.append(fifty_list, fifty_percent)
            eighty_list = np.append(eighty_list, eighty_percent)
            ninety_list = np.append(ninety_list, ninety_percent)

            #add each test set cube to a list over each uncertainty run
            pred_list[:, :, :, i] = pred_cube
            diff_list[:, :, :, i] = diff_cube
            mae2d_list[:, :, :, i] = mae_cube
            label_list[:, :, :, i] = label_cube
            img_list[:, :, :, i] = timex_cube

        #calculate stats over the test set ensemble runs
        modelmae = np.mean(mae_list)
        mae_err = 2*np.std(mae_list)
        modelrms = np.mean(rms_list)
        rms_err = 2*np.std(rms_list)
        modelgreatesterror = np.mean(greatest_list)
        greatest_var = 2*np.std(greatest_list)
        modeltenerror = np.mean(ten_list)
        ten_var = 2*np.std(ten_list)
        modelfiftyerror = np.mean(fifty_list)
        fifty_var = 2*np.std(fifty_list)
        modelsixtyerror = np.mean(sixtyfive_list)
        sixty_var = 2*np.std(sixtyfive_list)
        modeleightyerror = np.mean(eighty_list)
        eighty_var = 2*np.std(eighty_list)
        modelninetyerror = np.mean(ninety_list)
        ninety_var = 2*np.std(ninety_list)
        print("Model median ae: ", modelmae)
        print("Model rmse: ", modelrms)
        print("Model 90 error: ", modelninetyerror)

        #average over each uncertainty run to find ensemble means
        imgmean = np.mean(img_list, axis=3)
        labelmean = np.mean(label_list[:, downsample_zeroline:, :], axis=3)
        predmean = np.mean(pred_list[:, downsample_zeroline:, :], axis=3)
        diffmean = np.mean(diff_list[:, :, :], axis=3)
        mae2dmean = np.mean(mae2d_list[:, :, :], axis=3)
        pred_err = 2*np.std(pred_list[:, downsample_zeroline:, :], axis=3)

        #display rms of each pixel over entire test set
        rms2dmean = np.power(np.sum(np.power(diffmean, 2), axis=-1) / test_size, .5)


        diff_histo = np.mean(np.mean(diffmean, axis=0), axis=0)
        rms_histo = np.power(np.sum(np.sum(np.power(diffmean, 2), axis=0), axis=0)/(bathy_rows*(bathy_cols-downsample_zeroline)), .5)
        mae_histo = np.mean(np.mean(mae2dmean, axis=0), axis=0)
        predmax = np.amax(pred_list[:, downsample_zeroline:, :, :], axis=-1)
        predmin = np.amin(pred_list[:, downsample_zeroline:, :, :], axis=-1)
        within_2d = np.where((predmax > labelmean) & (labelmean > predmin), 1, 0)
#        l = 0
#        while l < test_size:
#            plt.imshow(within_2d[:, :, l])
#            plt.show()
#            l+=10
        print("within %: ", np.sum(within_2d)/(bathy_rows*bathy_cols*test_size))
        predict.plot(imgmean, labelmean, predmean, diffmean, rms2dmean, mae2dmean, pred_err, diff_histo, rms_histo, pred_list)

        imageio.mimsave('./' + name + '.gif', [plot_for_gif(imgmean, labelmean, predmean, diffmean, i, pred_list) for i in range(test_size)], fps=2.5)

    def calc_stats(self, imgdatas, imglabels):
        #load data to write
        imgs = imgdatas
        preds = np.load('./results/mask_test.npy')
        labels = imglabels

        #for each image in preds
        meanmae_list = []
        runrms_list = []
        greatest_error_list = []
        ten_percent_list = []
        twentyfive_percent_list = []
        fifty_percent_list = []
        seventyfive_percent_list = []
        ninety_percent_list = []

        timex_cube = np.zeros((bathy_rows, bathy_cols,  np.size(preds, 0)))
        label_cube = np.zeros((bathy_rows, bathy_cols, np.size(preds, 0)))
        pred_cube = np.zeros((bathy_rows, bathy_cols, np.size(preds,0)))
        diff_cube = np.zeros((bathy_rows, bathy_cols-downsample_zeroline, np.size(preds, 0)))
        mae_cube = np.zeros((bathy_rows, bathy_cols-downsample_zeroline, np.size(preds, 0)))
        i=0
        while i < len(preds):
            #grab first image, prediction image, and label
            img = imgs[i]
            img = img[:, :, :-1]
            img = img[:, :img_cols,::-1]
            img = np.mean(img, axis=2)
            img = cv2.resize(img, (bathy_cols, bathy_rows), interpolation=cv2.INTER_AREA)
            #img = np.expand_dims(img, axis=-1)
            pred = preds[i]
            #each cell is about 1.6m in size
            #resize so each cell is 5m in size which is resolution of measured bathy, for mae and uncertainty
            #1.6/5 = .32, so resize 512, 268 by .32 = (164, 86)
            pred = pred[:, :img_cols, 0]
            pred = cv2.resize(pred, (bathy_cols, bathy_rows), interpolation=cv2.INTER_AREA)
            #pred = cv2.blur(pred, (30, 30))
            pred[:, :downsample_zeroline] = 0

            label = labels[i]
            label = label[:, :img_cols,0]
            label = cv2.resize(label, (bathy_cols, bathy_rows), interpolation=cv2.INTER_AREA)
            label[:, :downsample_zeroline] = 0

            #experimental
            pred = np.where(label == 0, 0, pred)
            pred = cv2.blur(pred, (10,10))
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
            mae = np.power(np.power((pred[:, downsample_zeroline:]-label[:, downsample_zeroline:]), 2), .5)
            rms = np.power(np.sum(np.power((pred[:, downsample_zeroline:]-label[:, downsample_zeroline:]), 2))*(1/(bathy_rows*(bathy_cols-downsample_zeroline))), .5)
            difference = pred[:, downsample_zeroline:] - label[:, downsample_zeroline:]

            diff_cube[:, :, i] = difference
            mae_cube[:, :, i] = mae

            meanmae = np.mean(mae)
            greatest_error = np.amax(np.absolute(difference))
            ten_percent = np.percentile(np.absolute(difference), 10)
            twentyfive_percent = np.percentile(np.absolute(difference), 65)
            fifty_percent = np.percentile(np.absolute(difference), 50)
            seventyfive_percent = np.percentile(np.absolute(difference), 80)
            ninety_percent = np.percentile(np.absolute(difference), 95)

            meanmae_list = np.append(meanmae_list, meanmae)
            runrms_list = np.append(runrms_list, rms)
            greatest_error_list = np.append(greatest_error_list, greatest_error)
            ten_percent_list = np.append(ten_percent_list, ten_percent)
            twentyfive_percent_list = np.append(twentyfive_percent_list, twentyfive_percent)
            fifty_percent_list = np.append(fifty_percent_list, fifty_percent)
            seventyfive_percent_list = np.append(seventyfive_percent_list, seventyfive_percent)
            ninety_percent_list = np.append(ninety_percent_list, ninety_percent)

            i+=1

        modelmae = np.mean(meanmae_list)
        modelrms = np.mean(runrms_list)
        modelgreatesterror = np.mean(greatest_error_list)
        modeltenerror = np.mean(ten_percent_list)
        modelsixtyerror = np.mean(twentyfive_percent_list)
        modelfiftyerror = np.mean(fifty_percent_list)
        modeleightyerror = np.mean(seventyfive_percent_list)
        modelninetyerror = np.mean(ninety_percent_list)

        return modelmae, modelrms, modeltenerror, modelfiftyerror, \
               modelsixtyerror, modeleightyerror, modelninetyerror, modelgreatesterror, \
               pred_cube, diff_cube, mae_cube, label_cube, timex_cube

    def plot(self, imgmean, labelmean, predmean, diffmean, rms2dmean, mae2dmean, pred_err, diff_histo, rms_histo, pred_list):
        if args.fullstats:
            mpl.rcParams['agg.path.chunksize'] = zeroline


            img1d = imgmean[:, downsample_zeroline:].flatten()
            pred1d = predmean.flatten()
            label1d = labelmean.flatten()
            uc1d = pred_err.flatten()
            mae1d = mae2dmean.flatten()
            diff1d = diffmean.flatten()

            i=0
            uc1d_9 = []
            mae1d_9 = []
            uc1d_8 = []
            mae1d_8 = []
            uc1d_7 = []
            mae1d_7 = []
            uc1d_6 = []
            mae1d_6 = []
            uc1d_5 = []
            mae1d_5 = []
            uc1d_4 = []
            mae1d_4 = []
            uc1d_3 = []
            mae1d_3 = []
            uc1d_2 = []
            mae1d_2 = []
            uc1d_1 = []
            mae1d_1 = []
            uc1d_0 = []
            mae1d_0 = []
            diff1d_0 = []
            diff1d_1 = []
            diff1d_2 = []
            diff1d_3 = []
            diff1d_4 = []
            diff1d_5 = []
            diff1d_6 = []
            diff1d_7 = []
            diff1d_8 = []
            diff1d_9 = []
            while i < len(uc1d):
                if (uc1d[i] > .0) & (uc1d[i] < .1):
                    uc1d_0.append(uc1d[i])
                    mae1d_0.append(mae1d[i])
                    diff1d_0.append(diff1d[i])
                elif (uc1d[i] > .1) & (uc1d[i] < .2):
                    uc1d_1.append(uc1d[i])
                    mae1d_1.append(mae1d[i])
                    diff1d_1.append(diff1d[i])
                elif (uc1d[i] > .2) & (uc1d[i] < .3):
                    uc1d_2.append(uc1d[i])
                    mae1d_2.append(mae1d[i])
                    diff1d_2.append(diff1d[i])
                elif (uc1d[i] > .3) & (uc1d[i] < .4):
                    uc1d_3.append(uc1d[i])
                    mae1d_3.append(mae1d[i])
                    diff1d_3.append(diff1d[i])
                elif (uc1d[i] > .4) & (uc1d[i] < .5):
                    uc1d_4.append(uc1d[i])
                    mae1d_4.append(mae1d[i])
                    diff1d_4.append(diff1d[i])
                elif (uc1d[i] > .5) & (uc1d[i] < .6):
                    uc1d_5.append(uc1d[i])
                    mae1d_5.append(mae1d[i])
                    diff1d_5.append(diff1d[i])
                elif (uc1d[i] > .6) & (uc1d[i] < .7):
                    uc1d_6.append(uc1d[i])
                    mae1d_6.append(mae1d[i])
                    diff1d_6.append(diff1d[i])
                elif (uc1d[i] > .7) & (uc1d[i] < .8):
                    uc1d_7.append(uc1d[i])
                    mae1d_7.append(mae1d[i])
                    diff1d_7.append(diff1d[i])
                elif (uc1d[i] > .8) & (uc1d[i] < .9):
                    uc1d_8.append(uc1d[i])
                    mae1d_8.append(mae1d[i])
                    diff1d_8.append(diff1d[i])
                elif uc1d[i] > .9:
                    uc1d_9.append(uc1d[i])
                    mae1d_9.append(mae1d[i])
                    diff1d_9.append(diff1d[i])
                else:
                    continue
                i+=1

            z = np.polyfit(pred1d, mae1d, 1)
            p = np.poly1d(z)
            fit = p(pred1d)

            intense_z = np.polyfit(img1d, mae1d, 1)
            intense_p = np.poly1d(z)
            intense_fit = intense_p(img1d)

            #figure 1
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

            norm = mpl.cm.colors.Normalize(vmax=rms2dmean.max(), vmin=0)
            ax1.imshow(rms2dmean, cmap='jet', vmax=rms2dmean.max(), vmin=0, extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
            ax1.set_title('b) Spatial RMSE', fontsize=20)
            ax1.set_anchor('W')
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1)
            cbar.set_label('(m)', fontsize=14)

            norm = mpl.cm.colors.Normalize(vmax=-np.mean(diffmean, axis=-1).min(), vmin=np.mean(diffmean, axis=-1).min())
            ax2.imshow(np.mean(diffmean, axis=-1), cmap='bwr', vmin=np.mean(diffmean, axis=-1).min(), vmax=-np.mean(diffmean, axis=-1).min(), extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
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

            #figure 2
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
            ## change color and linewidth of the whiskers
            for whisker in bp['whiskers']:
                whisker.set(color='blue', linewidth=2)

            ## change color and linewidth of the caps
            for cap in bp['caps']:
                cap.set(color='blue', linewidth=2)

            medians = []
            ## change color and linewidth of the medians
            for median in bp['medians']:
                median.set(color='green', linewidth=2)
                medianY = []
                for j in range(2):
                    medianY.append(median.get_ydata()[j])
                medians.append(medianY[0])
            ## change the style of fliers and their fill
            for flier in bp['fliers']:
                flier.set(marker='o', color='red', alpha=0.3)

            #predmean = np.absolute(np.mean(pred_list[:, downsample_zeroline:, :], axis=3))
            plt.show()

        if args.indiv_pred_show:
            i = 0
            while i < test_size:
                plot_for_gif(imgmean, labelmean, predmean, diffmean, i, pred_list)
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
                print(np.shape(predmean[:, downsample_zeroline:, j]))
                X = np.linspace(zeroline, crosshore_distance_meters, crosshore_distance_meters-zeroline)
                Y = np.linspace(0, alongshore_distance_meters, alongshore_distance_meters)
                fig = plt.figure()
                fig.add_subplot(2, 2, 1), plt.imshow(imgmean[:, downsample_zeroline:, j], cmap='Greys_r', extent=[zeroline, crosshore_distance_meters, 0, alongshore_distance_meters])
                plt.title('a) Timex', fontsize=10)
                plt.tick_params(labelsize=10)
                plt.ylabel('Alongshore (m)', fontsize=10)
                fig.add_subplot(2, 2, 2), plt.imshow(-predmean[:, downsample_zeroline:, j], cmap='gist_earth', vmin=-6, vmax=4, extent=[zeroline,crosshore_distance_meters,0,alongshore_distance_meters])
                cbar = plt.colorbar()
                cbar.set_label('Elevation (m)', fontsize=10)
                plt.contour(X, Y, np.flip(-cv2.resize(predmean[:, downsample_zeroline:, j], (crosshore_distance_meters- zeroline, alongshore_distance_meters), interpolation=cv2.INTER_CUBIC),axis=0), colors='white', vmin=-6, vmax=4)
                plt.title('b) Ensemble Prediction', fontsize=10)
                plt.tick_params(labelsize=10)
                fig.add_subplot(2, 2, 3), plt.imshow(mae2dmean[:, :, j],cmap='jet', vmin=0, vmax=1, extent=[zeroline, crosshore_distance_meters, 0,alongshore_distance_meters])
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

    args = parser.parse_args()
    predict = Predictor()
    predict.eval()
