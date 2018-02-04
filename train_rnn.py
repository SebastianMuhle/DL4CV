import numpy
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

learning_data_root = 'data/learning/'
models_root = learning_data_root+'models/'
photo_root = learning_data_root + 'photos/'
train_photos_root = photo_root + 'train/'
validation_photos_root = photo_root + 'validation/'


epoch_size = 20
loop_in_epoch = 25000

filename = learning_data_root+"raw_review.txt"
raw_text = open(filename, encoding='utf-8').read()

chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

# Hyperparameters
lr = 0.01
dropoutRate = 0.4
hiddenDim = 256

print(lr, dropoutRate, hiddenDim)
# Create the optimizer
optimizerAdam = tf.keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0,)

# define the LSTM model
model = Sequential()
model.add(LSTM(hiddenDim, input_shape=(100, 1), return_sequences=True))
model.add(Dropout(dropoutRate))
model.add(LSTM(hiddenDim,return_sequences=True))
model.add(Dropout(dropoutRate))
model.add(LSTM(hiddenDim,return_sequences=True))
model.add(Dropout(dropoutRate))
model.add(LSTM(hiddenDim))
model.add(Dropout(dropoutRate))					
model.add(Dense(191, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer=optimizerAdam)

# define the checkpoint #get the hyperparamters info into the saveString (see my save string func in
# utility
filepath=models_root+"rnn_2.hdf5"

# prepare the dataset of input to output pairs encoded as integers
seq_length = 100 # maybe change it
X = []
y = []
loop_count = 500000
print("Loops: ",loop_count)
for i in range(0, loop_count, 1):
	if i % 100 == 0:
		print(i)
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	X.append([char_to_int[char] for char in seq_in])
	y.append(char_to_int[seq_out])
n_patterns = len(X)
print("Total Patterns: ", n_patterns)

char_to_int = None
int_to_char = None

# reshape X to be [samples, time steps, features]
X = numpy.reshape(numpy.array(X), (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(y)

checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=200, batch_size=100, callbacks=callbacks_list)