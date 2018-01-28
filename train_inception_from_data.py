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


train_data_root = 'data/train/'
val_data_root = 'data/validation/'

# Image Parameters
# Xception 299, 299 - VGG16 224, 224
img_width, img_height = 299, 299
img_shape = (img_width,img_height,3)

#DL Parameters
batch_size = 16

classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
nb_classes = 9

f = h5py.File(train_data_root+"inception_learning_data.hdf5","r")
x_learning = f['x_learning'][:]
y_learning = f['y_learning'][:]

# split train in train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x_learning, y_learning, test_size=0.25, random_state=42)

print(x_train.shape)
print(y_train.shape)
print(x_validation.shape)
print(y_validation.shape)

# Training data
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=utility.preprocess_input)

utility.apply_mean(train_datagen)

training_generator = train_datagen.flow(x_train,
								y_train,
								batch_size = batch_size)

# Validation data
validation_datagen = ImageDataGenerator(preprocessing_function=utility.preprocess_input)

utility.apply_mean(validation_datagen)

validation_generator = validation_datagen.flow(x_validation,
										y_validation,
										batch_size = batch_size)

# Hyperparameters
num_freezed_layers_array =[5, 80, 249]
learning_rates = [0.01, 0.001, 0.0001]

# Hyperparameter search
for num_freezed_layers in num_freezed_layers_array:
    for lr in learning_rates:

        save_string = utility.save_string(num_freezed_layers, lr)
        print(save_string)

        # Create Optimizer
        optimizerSGD = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

        # Create model
        model = InceptionV3model().create_model(num_freezedLayers=num_freezed_layers, nb_classes=nb_classes,
                                                optimizer=optimizerSGD)

        tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
                                    write_grads=False, write_images=False,
                                    #embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None
                                        )

        model.fit_generator(training_generator,
                            steps_per_epoch=1,  # nb_train_samples,
                            epochs=1,
                            validation_data=validation_generator,
                            validation_steps=1  # nb_validation_samples,
                            )

        # and predict on the test set
        predictions = model.predict_classes(test)

        model.save(save_string)
        model.save_weights("weights" + save_string)
