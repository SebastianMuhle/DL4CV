import tensorflow as tf
import utility
import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from .InceptionV3model import InceptionV3model
from .XceptionModel import XceptionModel

def apply_mean(image_data_generator):
    """Subtracts the dataset mean"""
    image_data_generator.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))

train_dir = 'test'
valid_dir = 'val'
img_width, img_height = 299, 299
batch_size = 16
classes = "" # to implement


# Training data
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function = utility.preprocess_input())
apply_mean(train_datagen)
training_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size = batch_size,
        classes = classes)

# Validation data
validation_datagen = ImageDataGenerator(preprocessing_function = utility.preprocess_input())
apply_mean(validation_datagen)
validation_data = validation_datagen.flow_from_directory(
        valid_dir,
        target_size=(img_width, img_height),
        batch_size = batch_size,
        classes = classes)


# Do to: Do following in a grid search  for hyper parameter tuning

# Hyperparameters
num_freezedLayersArray=[5, 80, 249]
learning_rates=[0.01, 0.001, 0.0001]

# Hyperparameter search
for num_freezedLayers in num_freezedLayersArray:
    for lr in learning_rates:

        # Create Optimizer
        optimizerSGD = tf.keras.optimizers.SGD(lr=lr, momentum=0.9)
        optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

        # Create model
        model = InceptionV3model().create_model(num_freezedLayers=num_freezedLayers, optimizer=optimizerSGD)

        tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,
                                    write_grads=False, write_images=False, embeddings_freq=0,
                                    embeddings_layer_names=None, embeddings_metadata=None)

        model.fit_generator(training_data,
                            steps_per_epoch=1,  # nb_train_samples,
                            epochs=1,
                            validation_data=validation_data,
                            validation_steps=1,  # nb_validation_samples,
                            )

        model.save("Model Inception num_freezedLayers %d lr %f" (num_freezedLayers, lr))
        model.sample_weights("Model Inception num_freezedLayers %d lr %f" (num_freezedLayers, lr))

