import tensorflow as tf
import train_vgg16_bottleneck
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras import layers


class Own_Classifier:

    def __init__(self):
        super(Own_Classifier, self).__init__()

    @staticmethod
    def create_model(nb_classes=9, noveltyDetectionLayerSize=1024, dropout = 0.5,
                     optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy'):
        model = tf.keras.models.Sequential()
        model.add(Flatten(input_shape=train_vgg16_bottleneck.train_data.shape[1:]))
        model.add(Dense(noveltyDetectionLayerSize, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(train_vgg16_bottleneck.num_classes, activation='sigmoid'))

        # compile model and return it
        model.compile(optimizer=optimizer, loss=loss)

        return model
