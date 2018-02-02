import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from RNNTextGeneration import RNNTextGeneration

predictions = np.array([1.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0])
textGenerator = RNNTextGeneration()
textGenerator.generateText(predictions,0.5,1000)
