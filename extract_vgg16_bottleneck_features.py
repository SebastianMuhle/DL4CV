import utility
import numpy as np
import tensorflow as tf
from VGG16_bottleneck import save_bottleneck_features
import pandas as pd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils import to_categorical
from InceptionV3model import InceptionV3model
from XceptionModel import XCeptionModel
from VGG16 import VGG16Model
from TopClassifier import TopClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import h5py
import math
from MultilabelGenerator import MultilabelGenerator


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
batch_size = 256
epoch_size = 5

classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
nb_classes = 9

#Get Data from HDF5 file
f = h5py.File(learning_data_root+"learning_data.hdf5","r")
x_train = f['x_train'][:]
y_train = f['y_train'][:]
x_validation = f['x_validation'][:]
y_validation = f['y_validation'][:]
# f2 = h5py.File(test_data_root+"test_data.hdf5","r")
# x_test = f2['x_test'][:]
# y_test = f2['y_test'][:]

train_photo_to_label_dict = dict(zip(x_train.tolist(),y_train.tolist()))
validation_photo_to_label_dict = dict(zip(x_validation.tolist(),y_validation.tolist()))
# test_photo_to_label_dict = dict(zip(x_test.tolist(),y_test.tolist()))

# Training data
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=utility.preprocess_input)

utility.apply_mean(train_datagen)

training_multilabel_datagen = MultilabelGenerator(photo_root,
                                    train_datagen,
                                    train_photo_to_label_dict,
                                    batch_size=batch_size,
                                    target_size=(img_width,img_height),
                                    train_or_valid='train')

training_generator = training_multilabel_datagen.flow()

names = [n.split('/')[-1].replace('.jpg','') for n in training_multilabel_datagen.directory_generator.filenames]
train_labels = np.zeros((len(names),nb_classes))
for i in range(len(names)):
    train_labels[i] = train_photo_to_label_dict[int(names[i])]


# Validation data
validation_datagen = ImageDataGenerator(preprocessing_function=utility.preprocess_input)

utility.apply_mean(validation_datagen)

validation_multilabel_datagen = MultilabelGenerator(photo_root,
                                    train_datagen,
                                    validation_photo_to_label_dict,
                                    batch_size=batch_size,
                                    target_size=(img_width,img_height),
                                    train_or_valid='validation')

validation_generator = validation_multilabel_datagen.flow()

names = [n.split('/')[-1].replace('.jpg','') for n in validation_multilabel_datagen.directory_generator.filenames]
validation_labels = np.zeros((len(names),nb_classes))
for i in range(len(names)):
    validation_labels[i] = validation_photo_to_label_dict[int(names[i])]

# Call function to extract VGG16 bottleneck features
print("Extracting Bottleneck Features")
save_bottleneck_features(training_generator,training_multilabel_datagen.directory_generator,validation_generator,validation_multilabel_datagen.directory_generator,batch_size,num_classes=nb_classes)

np.save(learning_data_root+'bottleneck_labels_training.npy', train_labels)
print("Saved bottleneck labels for training")
np.save(learning_data_root+'bottleneck_labels_validation.npy', validation_labels)
print("Saved bottleneck labels for validation")

