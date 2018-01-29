import utility
import numpy as np
import pandas as pd
from PIL import Image
import h5py
import os

learning_data_root = 'data/learning/'
photo_root = learning_data_root + 'photos/'
train_photos_root = photo_root + 'train/'
validation_photos_root = photo_root + 'validation/'

f = h5py.File(learning_data_root+"learning_data.hdf5","r")
x_train = f['x_train'][:]
y_train = f['y_train'][:]
x_validation = f['x_validation'][:]
y_validation = f['y_validation'][:]

print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)

print("Organizing Training Photos")
for i in range(len(x_train)):
	img_url = photo_root + str(x_train[i]) + '.jpg'
	if os.path.exists(img_url):
		new_img_url = train_photos_root + str(x_train[i]) + '.jpg'
		os.rename(img_url,new_img_url)
	if i % 100 == 0:
		print(i)

print("Organizing Validation Photos")
for i in range(len(x_validation)):
	img_url = photo_root + str(x_validation[i]) + '.jpg'
	if os.path.exists(img_url):
		new_img_url = validation_photos_root + str(x_validation[i]) + '.jpg'
		os.rename(img_url,new_img_url)
	if i % 100 == 0:
		print(i)

# print("Deleting Photos with no Business Id")