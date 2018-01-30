import utility
import numpy as np
import tensorflow as tf
from InceptionV3_bottleneck import save_bottleneck_features
import pandas as pd
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras.utils import to_categorical
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
img_width, img_height = 299, 299
img_shape = (img_width,img_height,3)

#DL Parameters
batch_size = 10
epoch_size = 10

classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']
nb_classes = 9

# Load extracted bottleneck features
train_data = np.load(learning_data_root+'inceptionv3_bottleneck_features_train.npy')
validation_data = np.load(learning_data_root+'inceptionv3_bottleneck_features_validation.npy')
train_labels = np.load(learning_data_root+'inceptionv3_bottleneck_labels_training.npy')
validation_labels = np.load(learning_data_root+'inceptionv3_bottleneck_labels_validation.npy')


# Hyperparameters
learning_rates = [0.005, 0.001, 0.0005, 0.0001]

# Hyperparameter search
for lr in learning_rates:

    save_string = utility.save_string(0, lr)
    utility.log(save_string)

    # Create Optimizer
    optimizerSGD = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
    optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

    model = TopClassifier().create_model(nb_classes=nb_classes, optimizer=optimizerAdam,input_shape=train_data.shape[1:])


    tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True,
                                    write_grads=False, write_images=False,
                                    #embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None
                                        )
    model.fit(train_data, train_labels,  
          epochs=epoch_size,  
          batch_size=batch_size,  
          validation_data=(validation_data, validation_labels))

    predictions = model.predict(train_data)

    accuracy_arr = np.zeros((train_data.shape[0]))
    for i in range(accuracy_arr.shape[0]):
        accuracy_arr[i] = f1_score(train_labels[i],np.around(np.asarray(predictions[i])))
    accuracy = np.mean(accuracy_arr)

    utility.log("Training - F1 Score: "+str(accuracy))
    utility.log("Training - Loss: "+str(model.evaluate(train_data,train_labels)))

    predictions = model.predict(validation_data)

    accuracy_arr = np.zeros((validation_data.shape[0]))
    for i in range(accuracy_arr.shape[0]):
        accuracy_arr[i] = f1_score(validation_labels[i],np.around(np.asarray(predictions[i])))
    accuracy = np.mean(accuracy_arr)

    utility.log("Validation - F1 Score: "+str(accuracy))
    utility.log("Validation - Loss: "+str(model.evaluate(validation_data,validation_labels)))

    model.save(save_string)
    model.save_weights(utility.save_weights_url(0, lr))
