from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# for model.fit function -> keras description
# class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value,
# used for weighting the loss function (during training only).
# This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
def get_class_weight(train_dir):
    # to be implemented
    class_weight = dict()
    return class_weight


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

