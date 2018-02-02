import utility
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import to_categorical
from InceptionV3model import InceptionV3model
from XceptionModel import XCeptionModel
from VGG16 import VGG16Model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PIL import Image
import h5py
from MultilabelGenerator import MultilabelGenerator


learning_data_root = 'data/learning/'
models_root = learning_data_root + 'models/'
photo_root = learning_data_root + 'photos/'
train_photos_root = photo_root + 'train/'
validation_photos_root = photo_root + 'validation/'

test_data_root = 'data/test/'
test_photo_root = learning_data_root + 'photos/'

# Image Parameters
# Xception 299, 299 - VGG16 224, 224
img_width, img_height = 299, 299
img_shape = (img_width,img_height,3)

#DL Parameters
batch_size = 64
epoch_size = 50

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

# Hyperparameters
num_freezed_layers_array =[132]
learning_rates = [0.001]

# Hyperparameter search
for num_freezed_layers in num_freezed_layers_array:
    for lr in learning_rates:

        save_string = utility.save_string(num_freezed_layers, lr)
        utility.log(save_string)

        # Create Optimizer
        optimizerSGD = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

        # Create model
        model = XCeptionModel().create_model(num_freezedLayers=num_freezed_layers, nb_classes=nb_classes,
                                                optimizer=optimizerAdam)

        tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
                                    write_grads=False, write_images=False,
                                    #embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None
                                        )

        filepath=models_root+"weights_xception.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

        model.fit_generator(training_generator,
                            steps_per_epoch=x_train.shape[0]/batch_size,  # nb_train_samples,
                            epochs=epoch_size,
                            verbose=1,
                            validation_data=validation_generator,
                            validation_steps=x_validation.shape[0]/batch_size
                            )

        model.load_weights(filepath)

        predict = model.predict_generator(training_generator, x_train.shape[0]/batch_size,verbose=1)

        accuracy_arr = np.zeros((len(training_multilabel_datagen.directory_generator.filenames)))
        for i, n in enumerate(training_multilabel_datagen.directory_generator.filenames):
            key = int(n.split('/')[-1].replace('.jpg',''))
            accuracy_arr[i] = f1_score(np.asarray(training_multilabel_datagen.photo_name_to_label_dict[key]),np.around(np.asarray(predict[i])))
        accuracy = np.mean(accuracy_arr)

        utility.log("Training - F1 Score: "+str(accuracy))
        utility.log("Training - Loss: "+str(model.evaluate_generator(training_generator, x_train.shape[0]/batch_size)))

        predict = model.predict_generator(validation_generator, x_validation.shape[0]/batch_size,verbose=1)

        accuracy_arr = np.zeros((len(validation_multilabel_datagen.directory_generator.filenames)))
        for i, n in enumerate(validation_multilabel_datagen.directory_generator.filenames):
            key = int(n.split('/')[-1].replace('.jpg',''))
            accuracy_arr[i] = f1_score(np.asarray(validation_multilabel_datagen.photo_name_to_label_dict[key]),np.around(np.asarray(predict[i])))
        accuracy = np.mean(accuracy_arr)

        utility.log("Validation - F1 Score: "+str(accuracy))
        utility.log("Validation - Loss: "+str(model.evaluate_generator(validation_generator, x_validation.shape[0]/batch_size)))
