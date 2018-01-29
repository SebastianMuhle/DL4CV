import utility
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py
import os
from PIL import Image

learning_data_root = 'data/learning/'
photo_root = learning_data_root + 'photos/'
train_photos_root = photo_root + 'train/'
validation_photos_root = photo_root + 'validation/'

test_data_root = 'data/test/'
test_photo_root = learning_data_root + 'photos/'

# Image Parameters
# Xception 299, 299 - VGG16 224, 224
img_width, img_height = 224, 224
img_shape = (img_width,img_height,3)

#DL Parameters
batch_size = 16

classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
nb_classes = 9

# csv to list in utility.py
business_labels = pd.read_csv(learning_data_root+'train.csv')
train_photo_business = pd.read_csv(learning_data_root+'train_photo_to_biz_ids.csv')

business_labels.dropna(inplace=True)

train = pd.merge(train_photo_business,business_labels, on="business_id")

train['file'] = photo_root + train['photo_id'].map(str) + '.jpg'

for i in range(len(classes)):
  train[classes[i]] = train['labels'].str.contains(str(i)).astype(int)

x_learning = np.asarray(train['photo_id'].tolist())
y_learning = np.asarray(train.ix[:,4:].values.tolist())

r_means = np.zeros((len(x_learning)))
g_means = np.zeros((len(x_learning)))
b_means = np.zeros((len(x_learning)))

for i in range(len(x_learning)):
	img_url = photo_root + str(x_learning[i]) + '.jpg'
	img = Image.open(img_url)
	img_array = np.asarray(img)
	r = img_array[:,:,0]
	r_means[i] = np.mean(r)
	g = img_array[:,:,1]
	g_means[i] = np.mean(g)
	b = img_array[:,:,2]
	b_means[i] = np.mean(b)
	if i %100 == 0 :
		print(i)
	
r_mean = np.mean(r)
g_mean = np.mean(g)
b_mean = np.mean(b)
	

