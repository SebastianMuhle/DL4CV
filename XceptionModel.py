import tensorflow as tf
from tensorflow.python.keras import layers


class XCeptionModel:

    def __init__(self):
        super(XCeptionModel, self).__init__()

    @staticmethod
    def create_model(num_freezedLayers=249, img_width=299, img_height=299, nb_classes=9, name_fclayer="fc1",
                     noveltyDetectionLayerSize=1024,
                     optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy'):

        # create the base pre-trained model
        base_model = tf.keras.applications.xception.Xception(weights = "imagenet",
                                                   include_top=False, input_shape = (img_width, img_height, 3))
        # add a global spatial average pooling layer
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(noveltyDetectionLayerSize, activation='relu', name = name_fclayer)(x)
        predictions = layers.Dense(nb_classes, activation='sigmoid')(x)

        # final model
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

        # freeze layers for transfer learning
        for layer in model.layers[:num_freezedLayers]:
            layer.trainable = False
        for layer in model.layers[num_freezedLayers:]:
            layer.trainable = True

        # compile model and return it
        model.compile(optimizer=optimizer, loss=loss)

        return model


