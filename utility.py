from tensorflow.python.keras.utils import to_categorical
import pandas as pd
import numpy as np
import datetime

model_data_root = 'data/learning/models/'
learning_data_root = 'data/learning/'

# for model.fit function -> keras description
# class_weight: Optional dictionary mapping class indices (integers) to a weight (float) value,
# used for weighting the loss function (during training only).
# This can be useful to tell the model to "pay more attention" to samples from an under-represented class.
def get_class_weight(train_dir):
    # to be implemented
    class_weight = dict()
    return class_weight


def preprocess_input(x):
    x -= 0.5
    x *= 2.
    return x.astype('float32')

def log(string):
    log_file = open(learning_data_root+"log.txt","a")
    print(string)
    log_file.write(string)

def save_string( num_freezed_layers, lr):
    current_date = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
    save_string1 =  model_data_root + current_date + "_VGG16_num_freezedLayers_%d _r_%g" % (num_freezed_layers, lr)
    save_string_return = save_string1.replace(" ", "_")
    save_string_return = save_string_return.replace(":", "_")
    save_string_return = save_string_return.replace(".", "_")
    save_string_return = save_string_return + ".h5"
    return save_string_return

def save_weights_url(num_freezed_layers, lr):
    current_date = '{:%d.%m.%Y %H:%M:%S}'.format(datetime.datetime.now())
    save_string1 =  model_data_root + "weights_" + current_date + "_VGG16_num_freezedLayers_%d _r_%g" % (num_freezed_layers, lr)
    save_string_return = save_string1.replace(" ", "_")
    save_string_return = save_string_return.replace(":", "_")
    save_string_return = save_string_return.replace(".", "_")
    save_string_return = save_string_return + ".h5"
    return save_string_return


def csv_to_lists(csv_file_name, sep=','):
    # parse csv
    df = pd.read_csv(csv_file_name, sep=sep)
    # change csv to list
    values_list = df.values.tolist()
    X = []
    y = []
    # parse training list and create x_train and y_train lists
    for element in values_list:
        X.append(element[0])
        # if that list does contain anything
        if type(element[1]) is str:
            y.append(list(map(int, element[1].split())))
        # if that list doesn't contain anything
        else:
            y.append([])

    X = np.array(X)
    y = np.array(y)
    return X, y


def to_multi_label_categorical(labels, dimension = 9):
    results = np.zeros((len(labels),dimension))
    for i in range(len(labels)):
        temp = to_categorical(labels[i],num_classes=dimension)
        results[i] = np.sum(temp, axis=0)
    return results

def apply_mean(image_data_generator):
    """Subtracts the dataset mean"""
    image_data_generator.mean = np.array([127.110, 105.946, 88.947], dtype=np.float32).reshape((3, 1, 1))
