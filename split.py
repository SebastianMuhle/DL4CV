import utility
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import h5py
import os

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
# test_photo_business = pd.read_csv(test_data_root+'test_photo_to_biz.csv')

business_labels.dropna(inplace=True)

train = pd.merge(train_photo_business,business_labels, on="business_id")
# test = pd.merge(test_photo_business,business_labels, on="business_id")
# test = pd.merge

train['file'] = photo_root + train['photo_id'].map(str) + '.jpg'
# test['file'] = test_photo_root + test['photo_id'].map(str) + '.jpg'

for i in range(len(classes)):
  train[classes[i]] = train['labels'].str.contains(str(i)).astype(int)
  # test[classes[i]] = test['labels'].str.contains(str(i)).astype(int)

x_learning = np.asarray(train['photo_id'].tolist())
y_learning = np.asarray(train.ix[:,4:].values.tolist())
# x_test = np.asarray(test['photo_id'].tolist())
# y_test = np.asarray(test.ix[:,4:].values.tolist())

print("Splitting Learnind Data to Train and Validation")

# split train in train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x_learning, y_learning, test_size=0.25, random_state=42)

# x_train = x_train[:50]
# y_train = y_train[:50]
# x_validation = x_validation[:10]
# y_validation = y_validation[:10]

print("Saving Data to HDF5 File")

f = h5py.File(learning_data_root+"learning_data.hdf5","w")
x_train_dataset = f.create_dataset("x_train", data=x_train)
y_train_dataset = f.create_dataset("y_train", data=y_train)
x_validation_dataset = f.create_dataset("x_validation", data=x_validation)
y_validation_dataset = f.create_dataset("y_validation", data=y_validation)
# f2 = h5py.File(test_data_root+"test_data.hdf5","w")
# x_test_dataset = f2.create_dataset("x_test", data=x_validation)
# y_test_dataset = f2.create_dataset("y_test", data=y_validation)

