#all the imports for every file
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch
from losses import *
from PIL import Image
from random import *
from scipy.io import loadmat
from scipy.stats import kde
from scipy import stats
from settings import *
from skimage.exposure import match_histograms
from torch.utils.data.dataset import Dataset  # For custom data-sets
from torch.utils.tensorboard import SummaryWriter
import argparse
import cv2
import glob
import imageio
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_scatter_density
import numpy as np
import pydot
import pywick
import scipy.interpolate as interp
import tensorflow.keras as keras
import tensorflow as tf
import tifffile as tif
import torch
import torch.nn as nn


# if using the one UAS image for test
def uas_handle():
    label_path = './z2017-02-28T00.15.00.Z'
    # not a real or duckgen image
    real = False
    duckgen = False
    UAS = True
    # load bathy from mat, resize to size of image
    label = loadmat(label_path)
    label = label['B']
    label = cv2.resize(label, (UAS_image_resize_height, UAS_image_resize_width),
                       interpolation=cv2.INTER_CUBIC)
    label = cv2.rotate(label, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
    new_label = np.zeros((512, 512))
    label = label[370:882, UAS_cell_offset:(UAS_cell_offset + img_cols)]
    new_label[:, :img_cols] = label
    label = new_label
    return real, duckgen, UAS, label


# if using histogram matching with input data
def histogram_match(g_source, seed):
    if seed > .0:
        g_reference = np.asarray(
            Image.open('./data/test/real2016/1452274201.Fri.Jan.08_17_30_01.GMT.2016.argus02b.cx.timex.merge.png'))
        if seed > .3:
            g_reference = np.asarray(
                Image.open('./data/test/real2016/1453491001.Fri.Jan.22_19_30_01.GMT.2016.argus02b.cx.timex.merge.png'))
            if seed > .5:
                g_reference = np.asarray(Image.open(
                    './data/test/real2016/1452709801.Wed.Jan.13_18_30_01.GMT.2016.argus02b.cx.timex.merge.png'))
                if seed > .7:
                    g_reference = np.asarray(Image.open(
                        './data/test/real2016/1477328401.Mon.Oct.24_17_00_01.GMT.2016.argus02b.cx.timex.merge.png'))
                    if seed > .9:
                        g_reference = np.asarray(Image.open(
                            './data/test/real2016/1483196401.Sat.Dec.31_15_00_01.GMT.2016.argus02b.cx.timex.merge.png'))
    g_matched = match_histograms(g_source, g_reference)
    image = g_matched.astype('int16')
    return image


# iadd gaussian noise to RGB input data before convert to grayscale
def gaussian_noise(g_source):
    noise = np.random.normal(0, noise_std, size=(512, 512, 3))
    image = g_source + noise
    return image


# add uniform noise across entire image to RGB input data
# before convert to grayscale
def uniform_noise(g_source):
    g_source[:, :, :, :-1] = g_source[:, :, :, :-1] / 255
    noise = np.random.normal(0, noise_std)
    image = g_source
    image[:, :, :, :-1] = g_source[:, :, :, :-1] + noise
    image = np.where(image < 0, 0, image)

    return image


# set hs, d, f according to WC lookup table
def set_cond(img_path):
    wc_index = img_path.find(".tif")
    wc = img_path[:wc_index]
    wc = wc[-2:]
    if wc[0] == '_':
        wc = wc[-1]
    if wc == '1':
        hs = 1.2
        d = 55
        f = .091
    elif wc == '2':
        hs = 1.2
        d = 55
        f = .125
    elif wc == '3':
        hs = 1.2
        d = 115
        f = .125
    elif wc == '4':
        hs = 2.3
        d = 55
        f = .167
    elif wc == '5':
        hs = 2.3
        d = 115
        f = .167
    elif wc == '6':
        hs = 2.3
        d = 55
        f = .091
    elif wc == '7':
        hs = 2.3
        d = 115
        f = .091
    elif wc == '8':
        hs = .7
        d = 80
        f = .091
    elif wc == '9':
        hs = 1.2
        d = 70
        f = .091
    elif wc == '10':
        hs = 1.7
        d = 60
        f = .140
    elif wc == '11':
        hs = 1.4
        d = 100
        f = .11
    elif wc == '12':
        hs = 1.1
        d = 84
        f = .131
    elif wc == '13':
        hs = 2.1
        d = 63
        f = .152
    elif wc == '14':
        hs = .9
        d = 108
        f = .116
    elif wc == '15':
        hs = 2.0
        d = 73
        f = .097
    elif wc == '16':
        hs = 1.5
        d = 94
        f = .149
    elif wc == '17':
        hs = 1.9
        d = 85
        f = .162
    elif wc == '18':
        hs = 1.6
        d = 63
        f = .114
    elif wc == '19':
        hs = 2.1
        d = 98
        f = .138
    elif wc == '20':
        hs = 1.8
        d = 112
        f = .169
    elif wc == '21':
        hs = 1.0
        d = 108
        f = .156
    elif wc == '22':
        hs = 2.2
        d = 78
        f = .118
    elif wc == '23':
        hs = 1.8
        d = 96
        f = .122
    elif wc == '24':
        hs = 0.8
        d = 76
        f = .146
    elif wc == '25':
        hs = 1.4
        d = 69
        f = .165
    elif wc == '26':
        hs = 0.9
        d = 99
        f = .168
    elif wc == '27':
        hs = 1.0
        d = 87
        f = .103
    elif wc == '28':
        hs = 1.6
        d = 106
        f = .101
    elif wc == '29':
        hs = 1.2
        d = 67
        f = .107
    elif wc == '30':
        hs = 1.8
        d = 56
        f = .109
    elif wc == '31':
        hs = 1.2
        d = 77
        f = .162
    elif wc == '32':
        hs = 2.0
        d = 71
        f = .114
    elif wc == '33':
        hs = 1.8
        d = 107
        f = .145
    elif wc == '34':
        hs = 1.2
        d = 86
        f = .149
    elif wc == '35':
        hs = 2.0
        d = 92
        f = .111
    elif wc == '36':
        hs = 2.1
        d = 67
        f = .123
    elif wc == '37':
        hs = 1.4
        d = 69
        f = .136
    elif wc == '38':
        hs = 0.8
        d = 108
        f = .137
    elif wc == '39':
        hs = 1.4
        d = 89
        f = .094
    elif wc == '40':
        hs = 1.8
        d = 98
        f = .099
    elif wc == '41':
        hs = 1.9
        d = 58
        f = .129
    elif wc == '42':
        hs = 0.8
        d = 104
        f = .127
    elif wc == '43':
        hs = 1.6
        d = 88
        f = .125
    elif wc == '44':
        hs = 1.2
        d = 103
        f = .164
    elif wc == '45':
        hs = 1.9
        d = 87
        f = .138
    else:
        hs = 0
        d = 0
        f = 0
    #normalize values between 0->1
    hs = (hs - .7) / (2.5 - .7)
    d = (d - 55) / (115 - 55)
    f = (f - 0.09) / (.18 - 0.09)
    return hs, d, f


# set the month index to search for bathy
def month_to_num(index, str_index):
    if index == 'Jan':
        index = 1
        str_index = '-01-'
    if index == 'Feb':
        index = 2
        str_index = '-02-'
    if index == 'Mar':
        index = 3
        str_index = '-03-'
    if index == 'Apr':
        index = 4
        str_index = '-04-'
    if index == 'May':
        index = 5
        str_index = '-05-'
    if index == 'Jun':
        index = 6
        str_index = '-06-'
    if index == 'Jul':
        index = 7
        str_index = '-07-'
    if index == 'Aug':
        index = 8
        str_index = '-08-'
    if index == 'Sep':
        index = 9
        str_index = '-09-'
    if index == 'Oct':
        index = 10
        str_index = '-10-'
    if index == 'Nov':
        index = 11
        str_index = '-11-'
    if index == 'Dec':
        index = 12
        str_index = '-12-'
        index = int(index)
    return index, str_index


# load either train or test files based on index
def train_or_test(self, idx):
    # idx is >test_id_offset if its test, derived from the flag given to unet.py/get_batch
    img_path = ''
    if idx < test_id_offset:
        test = False
        if real_or_fake == 'fake':
            img_path = self.generated_training[idx]
        if real_or_fake == 'real':
            temp = len(self.argus_training) - idx -1
            img_path = self.argus_training[temp]
    if idx >= test_id_offset:
        test = True
        idx = idx - test_id_offset
        if real_or_fake == 'fake':
            img_path = self.test_generated[idx]
        if real_or_fake == 'real':
            temp = len(self.test_observed)-idx-1
            img_path = self.test_observed[temp]
    return img_path, idx, test


# load either a synthetic, duckgen, or measured bathy and crop/resize
def find_bathy(self, img_path, idx):

    real = False
    duckgen = False
    bathy_index = img_path.find("_bathy_")

    # if bathy is not in filename, means either rbathy or real image
    if bathy_index == -1:
        duckgen = True
        bathy_index = img_path.find("_rbathy_")
        #if not even rbathy then must be real image
        if bathy_index == -1:
            real = True
            str_index = ''
            index = img_path[-52:-49]
            year = img_path[-32:-28]
            index, str_index = month_to_num(index, str_index)
            index = int(index)
        else:
            index = img_path[:bathy_index]
            index = index[-3:]
            if index[0] == '\\':
                index = index[1:]
            if index[1] == '\\':
                index = index[2]
            index = int(index)
    # if its bathy just load it up
    else:
        index = img_path[:bathy_index]
        index = index[-3:]

        if index[0] == '\\':
            index = index[1:]
        if index[1] == '\\':
            index = index[2]
        index = int(index)

    # crop and interpolation settings for real duckgen and synthetic images, see load_image for more comments
    if real:
        label_path = [i for i in self.measured_bathy if year in i]
        label_path = [i for i in label_path if str_index in i]
        cycle = True
        while cycle:
            try:
                label_path = label_path[0]
                cycle = False
            except:
                label_path, index = cycle_bathyno(self, year, index)
        if type(label_path) == list:
            label_path = label_path[0]
        label = loadmat(label_path)
        label = label['B']
        #label grid is 5x5m so cut off dimensions to make same size as argus (500mx1500m) so (100cellsx300cells)
        #hardcoded in to have shoreline line up with argus imagery
        label = label[12:112, 5:-17]
        label = cv2.resize(label, (real_bathy_resize_height, real_bathy_resize_width), interpolation=cv2.INTER_CUBIC)
        label = cv2.rotate(label, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_label = np.zeros((512, 512))
        south = north_bound[idx] + img_cols
        label = label[north_bound[idx]:south, real_cell_offset:(real_cell_offset+img_cols)]
        label = cv2.resize(label, (img_cols, img_rows))
        new_label[:, :img_cols] = label
        label = new_label
    elif duckgen:
        label_path = self.duckgen_bathy[index]
        label = loadmat(label_path)
        label = label['B']
        label = cv2.resize(label, (gen_image_resize_height, gen_image_resize_width), interpolation=cv2.INTER_CUBIC)
        label = cv2.rotate(label, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_label = np.zeros((512, 512))
        north = north_bound[idx] - 200 #subtract length difference from real image (2000->1805 for gen tiff)
        if north < 0:
            north = 0
        south = north + img_rows
        label = label[north:south, duckgen_offset:(duckgen_offset + img_cols)]
        new_label[:, :img_cols] = label
        label = new_label
    else:
        label_path = self.synthetic_bathy[index]
        label = loadmat(label_path)
        label = label['B']
        label = cv2.resize(label, (gen_image_resize_height, gen_image_resize_width), interpolation=cv2.INTER_CUBIC)
        label = cv2.rotate(label, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_label = np.zeros((512, 512))
        north = north_bound[idx] - 200 #subtract length difference from real image (2000->1805 for gen tiff)
        if north < 0:
            north = 0
        south = north + img_rows
        label = label[north:south, synthetic_offset:(synthetic_offset+img_cols)]
        new_label[:, :img_cols] = label
        label = new_label

    # don't use any values above 0, so slope of shore cannot be used during training or testing
    label = np.where((label > 0), 0, label)
    return label, real, duckgen


# if no bathy in same month as sample, look back 1 month
def cycle_bathyno(self, year, index):
    label_path = [i for i in self.measured_bathy if year in i]
    if index < 10:
        nindex = index - 1
        index = nindex
        nindex = '-0' + str(nindex) + '-'
    elif index > 9:
        nindex = index - 1
        index = nindex
        nindex = '-' + str(nindex) + '-'
    label_path = [i for i in label_path if nindex in i]
    return label_path, index


# load tif image, histogram equalize to real if fake, and crop/resize to input into network
def load_image(img_path, real, UAS, duckgen, idx, test, issnap):

    # load tiff, else load png
    try:
        image = tif.imread(img_path)
        if issnap:
            image = image[:, :img_cols, :]
    except:
        image = np.asarray(Image.open(img_path))

    # crop & interpolation for argus imagery to randomly select a section and get to 1 cell -> 1m resolution
    if real:
        image = cv2.resize(image, (real_image_resize_height, real_image_resize_width), interpolation=cv2.INTER_CUBIC)
        image = cv2.rotate(image, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_image = np.zeros((512, 512, 3))
        south = north_bound[idx] + img_rows
        new_image[:, :img_cols, :] = image[north_bound[idx]:south, real_image_offset:(real_image_offset+img_cols), :3]
        image = new_image

    # crop & interpolation for UAS imagery to randomly select a section and get to 1 cell -> 1m resolution
    elif UAS:
        image = cv2.resize(image, (UAS_image_resize_width, UAS_image_resize_height), interpolation=cv2.INTER_CUBIC)
        new_image = np.zeros((512, 512, 3))
        new_image[:, :img_cols, :] = image[200:712, UAS_image_offset:(UAS_image_offset+img_cols), :3]
        image = new_image
        g_source = image
        g_reference = tif.imread('./data/train/timex/13_rbathy_WC_1.tiff')
        g_matched = match_histograms(g_source, g_reference)
        image = g_matched.astype('int16')

    # crop & interpolation settings for synthetic imagery of duck,nc to randomly select a section and
    # get to 1 cell -> 1m resolution, also histogram matching with a random argus image
    elif duckgen:
        new_image = np.zeros((512, 512, 3))
        north = north_bound[idx] - 200
        if north < 0:
            north = 0
        south = north + img_rows
        image = np.array(image, dtype='uint8')
        image = cv2.resize(image, (gen_image_resize_width, gen_image_resize_height), interpolation=cv2.INTER_CUBIC)
        image = image[north:south, duckgen_offset:(duckgen_offset+img_cols)]
        new_image[:, :img_cols, :] = image
        image = new_image

    # crop & interpolation settings for completely synthetic imagery to randomly select a section and
    # get to 1 cell -> 1m resolution, also histogram matching with a random argus image
    else:
        new_image = np.zeros((512, 512, 3))
        north = north_bound[idx] - 200
        if north < 0:
            north = 0
        south = north + img_rows
        try:
            image = np.array(image, dtype='uint8')
            image = cv2.resize(image, (gen_image_resize_width, gen_image_resize_height), interpolation=cv2.INTER_CUBIC)
            image = image[north:south, synthetic_offset:(synthetic_offset+img_cols)]
            new_image[:, :img_cols, :] = image
            image = new_image
        except:
            print(img_path)
            plt.imshow(image)
            plt.show()

    image = np.mean(image, axis=2)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)

    return image


# load tif snap in same method as load_image
def load_snap(img_path, real, UAS, duckgen, idx, test):

    #change timex in folder to snap
    img_path = list(img_path)
    img_path[7:30] = 'all_snap/'
    #img_path = np.delete(img_path, 14)
    img_path = "".join(img_path)
    issnap=True
    #load snap w/ same method used to load timex image
    snap = load_image(img_path, real, UAS, duckgen, idx, test, issnap)
    return snap


# add channel to input sample of slope, hs, d, f
def add_channel(label, hs, d, f):
    # find the -.01 farthest to the right for each row
    slopeindex = np.sum(np.any(label > -.01, axis=0))
    shoreline_elevation = np.mean(label[:, slopeindex])
    offshore_elevation = np.mean(label[:, (img_cols-1)])
    #divide by constant img_cols instead of (img_cols-slopeindex) to introduce noise into the slope "guess"
    shoreslope = (shoreline_elevation - offshore_elevation) / (img_cols)
    #apply a stretch of 10 to get values closer to median of .5 of other addtl inputs
    shoreslope = shoreslope*10
    #shoreslope = 0
    # fill shoreline with 0s
    label[:, :zeroline] = 0

    #fill channel with shoreslope value
    labelslope = np.full(label.shape, shoreslope)
    #optionally add offshore wave conditions to each quadrant
    #labelslope[:256, 256:] = d
    #labelslope[256:, 256:] = hs
    #labelslope[256:, :256] = f
    labelslope = np.expand_dims(labelslope, axis=0)
    labelslope = np.expand_dims(labelslope, axis=-1)
    return labelslope

# dataset class
class TimexDataset(Dataset):
    def __init__(self, transform=None):
        self.generated_training = (glob.glob('./data/train/duckgen+syn/timex/*.tiff'))
        self.argus_training = sorted(glob.glob('./data/train/real_2015_2017/timex/*.png'))

        self.synthetic_bathy = glob.glob('./data/labels/*.mat')
        self.duckgen_bathy = glob.glob('./data/labels/duckgen_bathy/*.mat')
        self.measured_bathy = glob.glob('./data/labels/measured_bathy/*.mat')

        self.test_generated = sorted(glob.glob('./data/test/fakediff/timex/*.tiff'))
        self.test_observed = sorted(glob.glob('./data/test/300_real_test/timex/*.png'))

        self.transform = transform

    def __getitem__(self, idx):
        return self.load_file(idx)

    def __len__(self):
        if real_or_fake == 'fake':
            return len(self.generated_training)
        if real_or_fake == 'real':
            return len(self.argus_training)

    def load_file(self, idx):
        self.synthetic_bathy.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.duckgen_bathy.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        self.measured_bathy.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        #use idx to identify to pull train or test data
        img_path, idx, test = train_or_test(self, idx)

        #if testing on the one UAV image we have:
        if img_path == './data/test\Combo_Timex.png':
            real, duckgen, UAS, label = uas_handle()
        else:
            UAS = False
            label, real, duckgen = find_bathy(self, img_path, idx)

        #find wave conditions from lookup table
        hs, d, f = set_cond(img_path)

        #load timex image
        image = load_image(img_path, real, UAS, duckgen, idx, test, issnap=False)

        #load snapshot image if desired
        if snap:
            snap_image = load_snap(img_path, real, UAS, duckgen, idx, test)

        #create an additional channel with slope, wave height direction and period information in each quadrant
        info_channel = add_channel(label, hs, d, f)

        #add snap to 2nd channel if desired
        if snap:
            image = np.concatenate((image, snap_image), axis=3)

        #add additional channel to image
        image = np.concatenate((image, info_channel), axis=3)

        #convert to float32
        image = np.full(image.shape, image, dtype='float32')

        #normalize the grayscale channels
        g_source = image
        seed = np.random.random()
        if test:
            # image[:, :, :, :-1] = image[:, :, :, :-1] / 255
            image = uniform_noise(g_source)
            # image = gaussian_noise(g_source)
            # image = histogram_match(g_source, seed)
        else:
            image[:, :, :, :-1] = image[:, :, :, :-1] / 255

        #add zeros in image/label from the shoreline according to zeroline setting
        image[:, :, :zeroline, :] = 0
        label[:, :zeroline] = 0

        #randomly flip inputs horizontally to increase training data
        randomseed = random()
        if (randomseed > .5) & (not test):
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=0)

        sample = {'image': image, 'label': label}

        #plot for visualization
        """print(img_path)
        fig = plt.figure()
        X = np.linspace(0, img_cols, img_cols)
        Y = np.linspace(0, img_rows, img_rows)
        cs_labels = ["-8m", "-7.5m", "-7m", "-6.5m", "-6m", "-5.5m", "-5m", "-4.5m", "-4m", "-3.5m", "-3m", "-2.5m", "-2m", "-1.5m", "-1m", "-.5m", "0m"]
        fmt = {}
        for l, s in zip([-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], cs_labels):
            fmt[l] = s
        ax0 = fig.add_subplot(1, 3, 1), plt.imshow(image[0, :, :img_cols, 0], cmap='Greys_r')
        cs = ax0[0].contour(X, Y, np.where(label[:, :img_cols] > .1, 0, label[:, :img_cols]), vmin=-6, vmax=2, alpha=.5,
                colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
        ax0[0].clabel(cs, [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], fmt=fmt, inline_spacing = 2, fontsize='small',)
        ax1 = fig.add_subplot(1, 3, 2), plt.imshow(image[0, :, :img_cols, 1], cmap='Greys_r')
        cs = ax1[0].contour(X, Y, np.where(label[:, :img_cols] > .1, 0, label[:, :img_cols]), vmin=-6, vmax=2, alpha=.5,
                colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
        ax1[0].clabel(cs, [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], fmt=fmt, inline_spacing = 2, fontsize='small',)
        ax2 = fig.add_subplot(1, 3, 3), plt.imshow(label[:, :img_cols], cmap='gist_earth', vmin=-6, vmax=1)
        cs = ax2[0].contour(X, Y, np.where(label[:, :img_cols] > .1, 0, label[: , :img_cols]), vmin=-6, vmax=2, alpha=1,
                colors=['white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'white', 'black'],
                levels=[-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01],
                linestyles=['solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid', 'dashed', 'solid'],
                linewidths=[1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 1.5, .5, 2])
        ax2[0].clabel(cs, [-8, -7.5, -7, -6.5, -6, -5.5, -5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, -.01], fmt=fmt, inline_spacing = 2, fontsize='small',)
        plt.show()"""""
        return sample
