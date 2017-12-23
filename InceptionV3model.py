import tensorflow as tf
from tensorflow.python.keras import layers

class InceptionV3model():

    def __init__(self):
        super(InceptionV3model, self).__init__()

    def create_model(self, num_freezedLayers=249, img_width=299, img_height=299, num_classes=9,name_fclayer="fc1",
                     optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss ='binary_crossentropy'):

        # create the base pre-trained model
        base_model = tf.keras.applications.inception_v3.InceptionV3(weights = "imagenet",
                                                   include_top=False, input_shape = (img_width, img_height, 3))
        # add a global spatial average pooling layer
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation='relu', name = name_fclayer)(x)
        predictions = layers.Dense(num_classes, activation='sigmoid')(x)

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
