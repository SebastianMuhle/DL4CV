import utility
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils import to_categorical
from InceptionV3model import InceptionV3model
from XceptionModel import XCeptionModel
from VGG16 import VGG16Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import h5py

# DO NOT USE THAT, BECAUSE IT CREATES A 5X Larger Data for Each Image

train_data_root = 'data/train/'
val_data_root = 'data/validation/'

# Image Parameters
# Xception 299, 299 - VGG16 224, 224
img_width, img_height = 224, 224
img_size = img_width * img_height * 3
img_shape = (img_width,img_height,3)

#DL Parameters
batch_size = 16

classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
nb_classes = 9

# csv to list in utility.py
train_business_labels = pd.read_csv(train_data_root+'train.csv')
train_photo_business = pd.read_csv(train_data_root+'train_photo_to_biz_ids.csv')

train_business_labels.dropna(inplace=True)

train = pd.merge(train_photo_business,train_business_labels, on="business_id")

train['file'] = train_data_root + 'photos/'+ train['photo_id'].map(str) + '.jpg'


for i in range(len(classes)):
  train[classes[i]] = train['labels'].str.contains(str(i)).astype(int)

photo_to_class = dict(zip(train['file'].tolist(),train.ix[:,4:].values.tolist()))

image_urls = train['file'].tolist()
# image_urls = image_urls[:100]
x_learning = np.zeros((len(image_urls), img_width,img_height,3), dtype=np.uint8)
y_learning = np.zeros((len(image_urls), nb_classes), dtype=np.uint8)

for i in range(len(image_urls)):
	img = Image.open(image_urls[i])
	img = img.resize((img_width,img_height))
	x_learning[i] = img_to_array(img)
	y_learning[i] = photo_to_class[image_urls[i]]
	if (i%100 == 0):
		print(i)
		

print(x_learning.shape)
print(y_learning.shape)

f = h5py.File(train_data_root+"inception_learning_data.hdf5","w")
x_set = f.create_dataset("x_learning", data=x_learning)
y_set = f.create_dataset("y_learning", data=y_learning)
	


