import tensorflow as tf
import numpy as np
import math

learning_data_root = 'data/learning/'

def save_bottleneck_features(training_generator,training_directory_generator,validation_generator,validation_directory_generator,batch_size,num_classes=9):
    # Training parameters for bottleneck features

    # Training features
    nb_train_samples = len(training_directory_generator.filenames)
    num_classes = num_classes
    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    # Validation features
    nb_validation_samples = len(validation_directory_generator.filenames)
    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    # Bottleneck extraction

    # Create VGG16 Base model for bottleneck feature extraction
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

    # Extract bottleneck features for training data
    bottleneck_features_train = model.predict_generator(
        training_generator, predict_size_train)
    np.save(learning_data_root+'vgg16_bottleneck_features_train.npy', bottleneck_features_train)

    print("Saved bottleneck features for training")

    # Extract bottleneck features for validation data
    bottleneck_features_validation = model.predict_generator(
        validation_generator, predict_size_validation)
    np.save(learning_data_root+'vgg16_bottleneck_features_validation.npy', bottleneck_features_validation)

    print("Saved bottleneck features for validation")

