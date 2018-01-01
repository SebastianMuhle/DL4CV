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

def save_string(selfs, num_freezed_layers, lr):
    current_date = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
    save_string = current_date + "_Inception_num_freezedLayers_%d _r_%g" % (num_freezed_layers, lr)
    save_string_return = save_string.replace(" ", "_")
    save_string_return = save_string_return.replace(":", "_")
    save_string_return = save_string_return.replace(".", "_")
    save_string_return = save_string_return +".h5"
    return save_string_return

