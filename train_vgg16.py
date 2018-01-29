import utility
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils import to_categorical
from InceptionV3model import InceptionV3model
from XceptionModel import XCeptionModel
from VGG16 import VGG16Model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import h5py


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
batch_size = 5
epoch_size = 20

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

training_generator = utility.multilabel_flow(photo_root,
                                    train_datagen,
                                    train_photo_to_label_dict,
                                    bs=batch_size,
                                    target_size=(img_width,img_height),
                                    train_or_valid='train')

# Validation data
validation_datagen = ImageDataGenerator(preprocessing_function=utility.preprocess_input)

utility.apply_mean(validation_datagen)

validation_generator = utility.multilabel_flow(photo_root,
                                    train_datagen,
                                    validation_photo_to_label_dict,
                                    bs=batch_size,
                                    target_size=(img_width,img_height),
                                    train_or_valid='validation')

# Hyperparameters
num_freezed_layers_array =[14,16,18,20]
learning_rates = [0.01,0.001,0.0001]

# Hyperparameter search
for num_freezed_layers in num_freezed_layers_array:
    for lr in learning_rates:

        save_string = utility.save_string(num_freezed_layers, lr)
        print(save_string)

        # Create Optimizer
        optimizerSGD = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

        # Create model
        model = VGG16Model().create_model(num_freezedLayers=num_freezed_layers, nb_classes=nb_classes,
                                                optimizer=optimizerAdam)

        tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
                                    write_grads=False, write_images=False,
                                    #embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None
                                        )

        model.fit_generator(training_generator,
                            steps_per_epoch=x_train.shape[0]/batch_size,  # nb_train_samples,
                            epochs=epoch_size,
                            verbose=2,
                            validation_data=validation_generator,
                            validation_steps=x_validation.shape[0]/batch_size,  # nb_validation_samples,
                            callbacks=[metrics]
                            )

        predict = model.predict_generator(training_generator, x_train.shape[0]/batch_size)
        f1_score(y_train,predict)

        accuracy = 0
        for i, n in enumerate(training_generator.filenames):
            accuracy += f1_score(train_photo_to_label_dict[n],predict[i])
        accuracy /= len(training_generator.filenames)
        print("F1 Score: ",accuracy)

        model.save(save_string)
        model.save_weights("weights" + save_string)
