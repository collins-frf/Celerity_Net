# -*- coding:utf-8 -*-
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import glob
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tif
from scipy.io import loadmat

class dataProcess(object):
	def __init__(self, out_rows, out_cols, train_path = "data/train",
	 fake_train_path="data/faketrain", val_path = "data/val",
	 fake_val_path="data/fakeval", test_path="data/test", 
	 fake_test_path="data/faketest", npy_path="data/npydata", 
	runtime_npy_path="data/npydata/all", img_type="tiff", label_type="mat",
	bands=6):
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.train_path = train_path
		self.fake_train_path = fake_train_path
		self.fake_val_path = fake_val_path
		self.fake_test_path = fake_test_path
		self.img_type = img_type
		self.val_path = val_path
		self.test_path = test_path
		self.npy_path = npy_path
		self.runtime_npy_path = runtime_npy_path
		self.label_type = label_type
		self.bands = bands


	def split_data(self, slope_array, deep_array, avg_array):

		#load labels using loadmat
		labelpath = sorted(glob.glob("./bathy/*.mat"))

		for i in labelpath:
			label = loadmat(i)
			label = label['B']      

			label = cv2.resize(label, (1075,575), interpolation=cv2.INTER_LINEAR)
			label = cv2.rotate(label, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)


			label2 = label[537:1049,:512]

			#find the 0 farthest to the right for each row
			slope2index = np.sum(np.any(label2>0, axis=0))
			shoreslope2 = (np.mean(label2[:,slope2index]) - np.mean(label2[:,-1]))/(512-slope2index)
			shoreslope2 = shoreslope2*1.68
			avg_depth2 = (np.mean(label2[:,slope2index:]))
			print("Slope B: ",shoreslope2)
			print("Max Depth B: ",np.min(label2))
			print("Avg Depth B: ",avg_depth2)
			slope_array = np.append(slope_array, shoreslope2)
			deep_array = np.append(deep_array, np.min(label2))
			avg_array = np.append(avg_array, avg_depth2)
			print(deep_array.shape)
			print(slope_array.shape)
			print(avg_array.shape)

		return slope_array, deep_array, avg_array

	def create_data(self):

		slope_array = []
		deep_array = []
		avg_array = []

		slope_array, deep_array, avg_array = self.split_data(slope_array, deep_array, avg_array)
		fig = plt.figure()
		num_bins = 25

		ax = fig.add_subplot(1, 3, 1), plt.hist(slope_array, num_bins, facecolor='blue', alpha=0.5)
		plt.title('Slopes')
		plt.ylabel('# of images')
		plt.xlabel('meters per meter')
		ax = fig.add_subplot(1, 3, 2), plt.hist(deep_array, num_bins, facecolor='red', alpha=0.5)
		plt.title('Max Depth')
		plt.ylabel('# of images')
		plt.xlabel('Metres')
		ax = fig.add_subplot(1, 3, 3), plt.hist(avg_array, num_bins, facecolor='green', alpha=0.5)
		plt.title('Average Depth')
		plt.ylabel('# of images')
		plt.xlabel('Metres')
		plt.show()

if __name__ == "__main__":

	mydata = dataProcess(512, 512)
	mydata.create_data()
