#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat, savemat
import cv2
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from data import *

def generate_planar_bathy(size):
    #generate planar bathy with constant slope from 0->1
    planar_bathy = np.zeros((194,361), dtype=np.double)
    for i in range(len(planar_bathy[:,0])):
        for j in range(len(planar_bathy[0,:])):
            planar_bathy[i,j] = (i)/len(planar_bathy[:,0])
    planar_bathy = planar_bathy
    planar_bathy = np.flip(planar_bathy, axis=0)

    return planar_bathy

def generate_parametric_bathy(size):
    #generate parametric beach slope based on h = Ax^2/3
    planar_bathy = np.zeros((194,361), dtype=np.double)
    for i in range(len(planar_bathy[:,0])):
        for j in range(len(planar_bathy[0,:])):
            planar_bathy[i,j] = ((i)**(2/3))/(np.random.randint(30)+15)
    planar_bathy = planar_bathy
    planar_bathy = np.flip(planar_bathy, axis=0)

    return planar_bathy

def generate_random_noise(size, mu, sigma):
    r_noise = np.random.normal(mu, sigma, [194,361])

    return r_noise

def generate_sandbar(size):
    #get size of bathy
    h, w = size
    #max si   ze of sandbar
    sandbar_width = np.random.randint(40) + 1

    #locate sandbar somewhere inside the bathy range, dont start it further out than max sandbar width
    low_thresh = np.random.randint(h-sandbar_width)
    high_thresh = low_thresh + np.random.randint(sandbar_width)

    #check that low_thresh < high_thresh
    if low_thresh > high_thresh:
        temp = low_thresh
        low_thresh = high_thresh
        high_thresh = temp

    #if sandbar is on the shore area, push it out 50 pixels (250m)
    #nah this is fine, higher shorelines are good
    low_thresh+=25
    high_thresh+=25

    sandbar = np.zeros(size, dtype=np.double)

    for i in range(h):
        if high_thresh < 160:
            if ((i > low_thresh) & (i < high_thresh)):
                for j in range(w):
                    sandbar[i,j] = np.random.random()

    return sandbar

def generate_trough(size):
    #get size of bathy
    h, w = size
    #max size of sandbar
    trough_width = np.random.randint(40) + 1

    #locate sandbar somewhere inside the bathy range, dont start it further out than max sandbar width
    low_thresh = np.random.randint(h-trough_width)
    high_thresh = low_thresh + np.random.randint(trough_width)

    #check that low_thresh < high_thresh
    if low_thresh > high_thresh:
        temp = low_thresh
        low_thresh = high_thresh
        high_thresh = temp

    #if trough is on the shore area, push it out 50 pixels (250m)
    low_thresh+=25
    high_thresh+=25

    trough = np.zeros(size, dtype=np.double)

    for i in range(h):
        if ((i > low_thresh) & (i < high_thresh)):
            for j in range(w):
                trough[i,j] = np.random.random()

    trough = -trough
    return trough

def generate_spot(size):
    #get size of bathy
    h, w = size
    #max size of sandbar
    spot_width = np.random.randint(40) + 1
    #locate sandbar somewhere inside the bathy range, dont start it further out than max sandbar width
    row_low_thresh = np.random.randint(h-spot_width+10)
    row_high_thresh = row_low_thresh + np.random.randint(spot_width)
    col_low_thresh = np.random.randint(w-spot_width)
    col_high_thresh = col_low_thresh + np.random.randint(spot_width)
    #check that low_thresh < high_thresh
    if row_low_thresh > row_high_thresh:
        temp = row_low_thresh
        row_low_thresh = row_high_thresh
        row_high_thresh = temp
    if col_low_thresh > col_high_thresh:
        temp = col_low_thresh
        col_low_thresh = col_high_thresh
        col_high_thresh = temp
    #if spot is deep in the shore area, push it out 50 pixels (250m)
    row_low_thresh+=25
    row_high_thresh+=25

    spot = np.zeros(size, dtype=np.double)

    for i in range(h):
        if ((i > row_low_thresh) & (i < row_high_thresh)):
            for j in range(w):
                if ((j > col_low_thresh) & (j < col_high_thresh)):
                    spot[i,j] = np.random.random()

    #randomly add or subtract spot
    randomseed = np.random.random()
    if (randomseed > .5):
        spot = -spot

    return spot

def plot_process(new_bathy, blurred_planarnoise, troughs, sandbars, spots, new_mat):
    print(np.shape(new_mat))
    X = np.linspace(0, 970, 970)
    Y = np.linspace(0, 1795, 1795)

    blurred_planarnoise = cv2.resize(blurred_planarnoise, (970, 1795), interpolation=cv2.INTER_CUBIC)
    new_mat = cv2.resize(new_mat, (970, 1795), interpolation=cv2.INTER_CUBIC)
    plt.subplot(1, 3, 1), plt.imshow(blurred_planarnoise, cmap='gist_earth',  extent=[0,970,1795,0])
    plt.title('a) Noisy Parametric Slope', fontsize=20)
    plt.ylabel('Distance Alongshore (m)', fontsize=16)
    plt.xlabel('Distance Crosshore (m)', fontsize=16)
    plt.tick_params(labelsize=14)
    #plt.contour(X, Y, blurred_planarnoise,  vmin=-6, vmax=4, extent=[0,970,1795,0], colors='black')
    perturbations = troughs + sandbars + spots
    plt.subplot(1, 3, 2), plt.imshow(perturbations, cmap='bwr', vmin=-1, vmax=1, extent=[0,970,1795,0])
    plt.title('b) Perturbations', fontsize=20)
    plt.xlabel('Distance Crosshore (m)', fontsize=16)
    plt.tick_params(labelsize=14)
    cbar = plt.colorbar()
    cbar.set_label('(m)', fontsize=16)
    plt.subplot(1, 3, 3), plt.imshow(new_mat, cmap='gist_earth', vmin=-6, vmax=4, extent=[0,970,1795,0])
    plt.title('c) Composite Bathymetry', fontsize=20)
    plt.xlabel('Distance Crosshore (m)', fontsize=16)
    cbar = plt.colorbar()
    cbar.set_label('Elevation (m)', fontsize=16)
    plt.contour(X, Y, np.where(new_mat > .1, 0, new_mat), extent=[0,970,1795,0], colors='white')
    plt.tick_params(labelsize=14)
    plt.show()

def plot_profiles(n, new_mat, profile, profile_hist, xdim):
    step_size = 1
    if n%step_size == 0:
        plt.subplot(1, 3, 1), plt.imshow(new_mat, cmap='gist_earth', vmin=-9, vmax=4)
        plt.title('Blurred Bathymetry', fontsize=36)
        plt.ylabel('y (m)', fontsize=20)
        plt.xlabel('x (m)', fontsize=20)
        plt.colorbar()
        plt.subplot(1, 3, 2), plt.plot(xdim, profile)
        plt.ylim(-8,4)
        plt.title('Bathymetric Profile', fontsize=36)
        plt.ylabel('y (m)', fontsize=20)
        plt.xlabel('x (m)', fontsize=20)
        plt.tick_params(labelsize=28)
        for profile in profile_hist:
            plt.subplot(1, 3, 3), plt.plot(xdim, profile, c='b')
            plt.ylim(-8,4)
            plt.title('Generated Profile Composite', fontsize=36)
            plt.ylabel('y (m)', fontsize=20)
            plt.xlabel('x (m)', fontsize=20)
            plt.tick_params(labelsize=28)
    plt.show()

def plot_profile_density(x_values, y_values, dx_values, dy_values, bins):
    x_values_0 = []
    y_values_0 = []
    dy_values_0 = []
    dx_values_0 = []
    i=0
    while i < len(x_values):
        if (y_values[i] < .0):
            x_values_0.append(x_values[i])
            y_values_0.append(y_values[i])
        i+=1
    i=0
    while i < len(dx_values):
        if (dy_values[i] < .0):
            dx_values_0.append(dx_values[i])
            dy_values_0.append(dy_values[i])
        i+=1
    fig=plt.figure()
    ax1 = fig.add_subplot(1,3,1), plt.hexbin(dx_values_0, dy_values_0, bins='log', cmap='jet', vmax=100)
    plt.title('a) Duck, NC profiles', fontsize=18)
    plt.xlabel('Cross-shore (m)', fontsize=16)
    plt.ylabel('Elevation (m)', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.grid(True)
    ax2 = fig.add_subplot(1,3,2), plt.hexbin(x_values_0, y_values_0, cmap='jet', bins='log', vmax=100)
    plt.title('b) Synthetic profiles', fontsize=18)
    plt.xlabel('Cross-shore (m)', fontsize=16)
    plt.tick_params(labelsize=14)
    norm = ImageNormalize(vmin=0, vmax=100, stretch=LogStretch())
    cmap = mpl.cm.jet
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap))
    cbar.set_label('Count', fontsize=14)
    plt.grid(True)
    plt.show()

def main():
    example_mat = loadmat('./bathy/celeris_bathy.mat')

    #number of bathys to generate
    bathy_no = 100
    profile_hist = []
    dprofile_hist = []
    y_values = []
    x_values = []
    dy_values = []
    dx_values = []
    for n in range(bathy_no):
        bathy_no+=30
        bathy_size = (194,361)
        new_bathy = np.zeros((194,361), dtype=np.double)

        #generate planar bathy
        planar_bathy = generate_parametric_bathy(bathy_size)

        #generate random noise
        mu, sigma = -.5, .2
        r_noise = generate_random_noise(bathy_size, mu, sigma)

        #sum planar bathy and noise
        planarnoise = planar_bathy + r_noise

        #smooth
        blurred_planarnoise = cv2.blur(planarnoise, (10,10))

        #generate random number of sandbars (at least 5)
        sandbars = np.zeros(bathy_size, dtype=np.double)
        sandbar_no = np.random.randint(10)+5
        for i in range(sandbar_no):
            sandbar = generate_sandbar(bathy_size)
            sandbars += sandbar
        #so that ones that generate on top of each other wont be higher than highest point
        #sandbars = sandbars/(sandbar_no)
        #blur to make more realistic
        sandbars = cv2.blur(sandbars, (40,10))

        #generate random number of trough (at least 5)
        troughs = np.zeros(bathy_size, dtype=np.double)
        trough_no = np.random.randint(10)+5
        for i in range(trough_no):
            trough = generate_trough(bathy_size)
            troughs += trough
        #so that ones that generate on top of each other wont be higher than highest point
        troughs = troughs/trough_no
        #blur to make more realistic
        troughs = cv2.blur(troughs, (40,10)) *5

        #generate random number of spots (at least 50)
        spots = np.zeros(bathy_size, dtype=np.double)
        spot_no = np.random.randint(50)+50
        for i in range(spot_no):
            spot = generate_spot(bathy_size)
            spots += spot
        #so that ones that generate on top of each other wont be higher than highest point
        #spots = spots/spot_no
        #blur to make more realistic
        spots = cv2.blur(spots, (30,30)) *3

        #create new bathy as combination of planarnoise, sandbars, troughs, and spots
        new_bathy = planarnoise + sandbars + troughs + spots
        for i in range(len(new_bathy[:,0])):
            for j in range(len(new_bathy[0,:])):
                    new_bathy[i,j] = new_bathy[i,j]*20*np.random.random() #stretch up to 20 meters in base height

        new_bathy = new_bathy - 5.2

        for i in range(len(new_bathy[:,0])):
            for j in range(len(new_bathy[0,:])):
                #add 1 meter of height to the beach
                if i < 50:
                    new_bathy[i,j]+=1
                #if anything offshore is greater than 0 make it less than 0
                if ((i > 50) & (new_bathy[i,j] > 0)):
                    new_bathy[i,j] = 0 - np.random.random()
                #if anything is less than -9 meters make it -9m
                if new_bathy[i,j] < -6:
                    new_bathy[i,j] = -9

        final_bathy = cv2.blur(new_bathy, (30,30))
        new_mat = example_mat
        new_mat['B'] = final_bathy

        savemat('./bathy/celeris_gen_bathy' + str(140+n) + '.mat', new_mat)

        new_bathy = cv2.rotate(new_bathy, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        blurred_planarnoise = cv2.rotate(blurred_planarnoise, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        troughs = cv2.rotate(troughs, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        sandbars = cv2.rotate(sandbars, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        spots = cv2.rotate(spots, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        new_mat['B'] = cv2.rotate(new_mat['B'], rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)

    #plot the bathymetry making process for visualziation
        if args.process:
            plot_process(new_bathy, blurred_planarnoise, troughs, sandbars, spots, new_mat['B'])

    #plot each profile of the bathymetry for manual checking of realism
        xdim = np.linspace(0,965,194)
        profile = np.mean(new_mat['B'], axis=0)
        profile_hist.append(profile)
        #plot_profiles(n, new_mat['B'], profile, profile_hist, xdim)

    #plot a density heatmap of each profile
        y_values.extend(profile)
        x_values.extend(xdim)

    for n in range(100):
        example_mat = loadmat('./bathy/spicer_bathy/celeris_gen_bathy' + str(140+ n) + '.mat')
        example_mat['B'] = cv2.rotate(example_mat['B'], rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        dprofile = np.mean(example_mat['B'], axis=0)
        dprofile_hist.append(dprofile)
        dy_values.extend(dprofile)
        dx_values.extend(xdim)

        #plot_profiles(n, example_mat['B'], dprofile, dprofile_hist)
    bins = (194, 10)
    if args.density:
        plot_profile_density(x_values, y_values, dx_values, dy_values, bins)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-pr', '--process', action='store_true',
                        help="print full stats and uncertainty of test set with N passes")
    parser.add_argument('-de', '--density', action='store_true',
                        help="print out examples of individual uncertainty")
    args = parser.parse_args()
    main()