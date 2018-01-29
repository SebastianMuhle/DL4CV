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

train_photos = os.listdir(train_photos_root)
validation_photos = os.listdir(validation_photos_root)

i = 0

for photo in train_photos:
	os.rename(os.path.join(train_photos_root, photo), os.path.join(photo_root, photo))
	i += 1
	if i % 100 == 0:
		print(i)

for photo in validation_photos:
	os.rename(os.path.join(validation_photos_root, photo), os.path.join(photo_root, photo))
	i += 1
	if i % 100 == 0:
		print(i)
