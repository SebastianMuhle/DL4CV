import utility
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils import to_categorical
from InceptionV3model import InceptionV3model
from XceptionModel import XCeptionModel
from VGG16 import VGG16Model
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



classes = ['good_for_lunch', 'good_for_dinner', 'takes_reservations', 'outdoor_seating', 'restaurant_is_expensive',
               'has_alcohol', 'has_table_service', 'ambience_is_classy', 'good_for_kids']

# csv to list in utility.py
x_learning, y_learning = utility.csv_to_lists('train.csv')


#TODO: The below code should be changed accordingly.



#test=pd.read_csv('../input/test.csv') # change it
# train=shuffle(train) maybe we shouldn't add to this, to make different results comparable!
# only use 50% of training set
# train = train[:int(train.shape[0]*0.5)]
# print(train)

# # Check and change the following part!
# labels = train['labels']                    # save the target column for later use
# train = train.drop(['labels'], axis=1)   # drop label column from data set
# colnames = list(train)                    # save the columnnames

# split train in train and validation
x_train, x_validation, y_train, y_validation = train_test_split(x_learning, y_learning, test_size=0.25, random_state=42)

# one hot encoding on labels - change it to multi labels!
# Haydar: Since to_categorical is for one class, I have impelemented a function for multi_label in utility.py
y_train = utility.to_multi_label_categorical(y_train)
y_validation = utility.to_multi_label_categorical(y_validation)

print(y_train)
print(y_validation)

# Xception 299, 299 - VGG16 224, 224
img_width, img_height = 299, 299
batch_size = 16

# Training data
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=utility.preprocess_input)

utility.apply_mean(train_datagen)
training_data = train_datagen.flow(
        x_train
        # ,target_size=(img_width, img_height)
        ,batch_size = batch_size
        # ,classes = y_train
        )

# Validation data
validation_datagen = ImageDataGenerator(preprocessing_function=utility.preprocess_input)
utility.apply_mean(validation_datagen)
validation_data = train_datagen.flow(
        x_validation
        # ,target_size=(img_width, img_height)
        ,batch_size = batch_size
        # ,classes = y_validation
        )

# Hyperparameters
num_freezed_layers_array =[5, 80, 249]
learning_rates = [0.01, 0.001, 0.0001]
nb_classes = 9

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

        tbCallBack = tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                    write_grads=False, write_images=False,
                                    #embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None
                                        )

        model.fit_generator(training_data,
                            steps_per_epoch=1,  # nb_train_samples,
                            epochs=1,
                            validation_data=validation_data,
                            validation_steps=1,  # nb_validation_samples,
                            )

        # and predict on the test set
        predictions = model.predict_classes(test)

        model.save(save_string)
        model.save_weights("weights" + save_string)
