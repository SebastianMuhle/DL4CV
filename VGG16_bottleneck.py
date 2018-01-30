import tensorflow as tf
import train_vgg16_bottleneck as train
import numpy as np
import math


def save_bottleneck_features():
    # Training parameters for bottleneck features

    # Training features
    nb_train_samples = len(train.training_generator.filenames)
    num_classes = len(train.training_generator.class_indices)
    predict_size_train = int(math.ceil(nb_train_samples / train.batch_size))

    # Validation features
    nb_validation_samples = len(train.validation_generator.filenames)
    predict_size_validation = int(math.ceil(nb_validation_samples / train.batch_size))

    # Bottleneck extraction

    # Create VGG16 Base model for bottleneck feature extraction
    model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')

    # Extract bottleneck features for training data
    bottleneck_features_train = model.predict_generator(
        train.training_generator, predict_size_train)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    # Extract bottleneck features for validation data
    bottleneck_features_validation = model.predict_generator(
        train.validation_generator, predict_size_validation)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

