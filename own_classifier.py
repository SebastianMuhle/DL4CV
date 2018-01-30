import tensorflow as tf
from tensorflow.python.keras.layers import Dropout, Flatten, Dense
from tensorflow.python.keras import layers


class Own_Classifier:

    def __init__(self):
        super(Own_Classifier, self).__init__()

    @staticmethod
    def create_model(nb_classes=9, noveltyDetectionLayerSize=1024, dropout = 0.5,
                     optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',input_shape=1):
        model = tf.keras.models.Sequential()
        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(noveltyDetectionLayerSize, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='sigmoid'))

        # compile model and return it
        model.compile(optimizer=optimizer, loss=loss)

        return model
