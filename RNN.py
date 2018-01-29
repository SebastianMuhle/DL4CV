import numpy
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import np_utils

filename = "NameOfOurFile"
raw_text = open(filename).read()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100 # maybe change it
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

# Hyperparameters
num_layers = [2, 4, 6]
learning_rates = [0.01, 0.001, 0.0001]
dropoutRates = [0.2, 0.4, 0.5]
hiddenDims = [256]

for num_layer in num_layers:
	for lr in learning_rates:
		for dropoutRate in dropoutRates:
			for hiddenDim in hiddenDims:

				# Create the optimizer
				optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

				# define the LSTM model
				model = Sequential()
				model.add(LSTM(hiddenDim, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
				model.add(Dropout(dropoutRate))
				for i in range(num_layer - 1):
					model.add(LSTM(hiddenDim, return_sequences=True))
					model.add(Dropout(dropoutRate))
				model.add(Dense(y.shape[1], activation='softmax'))
				model.compile(loss='categorical_crossentropy', optimizer=optimizerAdam)

				# define the checkpoint #get the hyperparamters info into the saveString (see my save string func in
				# utility
				filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
				checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
				callbacks_list = [checkpoint]

				model.fit(X, y, epochs=20, batch_size=128, callbacks=callbacks_list)