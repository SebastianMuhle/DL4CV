from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.callbacks import ModelCheckpoint
import numpy as np
import sys


class RNNTextGeneration:

    learning_data_root = 'data/learning/'
    models_root = learning_data_root+'models/'

    # Same as in the RNN function, has to be changed afterwards
    filename = learning_data_root+"raw_review.txt"
    raw_text = open(filename, encoding='utf-8').read()
    chars = sorted(list(set(raw_text)))

    # Info to check if we loaded the right file
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    int_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    dropoutRate = 0.4
    hiddenDim = 256

    # load the network weights
    filename = models_root+"rnn.hdf5"
    model = Sequential()
    model.add(LSTM(hiddenDim, input_shape=(100, 1), return_sequences=True))
    model.add(Dropout(dropoutRate))
    model.add(LSTM(hiddenDim))
    model.add(Dropout(dropoutRate))                 
    model.add(Dense(191, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.load_weights(filename)

    # Prediction text pieces
    predictionTextPieces = ["This place is really good for lunch. We had a great time. I would prefer everyone to lunch in this  ",
                            "This place is really great for dinner. Food was really delicious. I would prefer this place to anyon",
                            "This place is a nice place with reservations. You don't wait outside, while it takes reservations   ",
                            "This place is amazing. It has outdoor seating where you can smoke. It has a good view from outdoor  ",
                            "This place is really nice, but it is unfortunately expensive. I would go there for special occasions",
                            "This place is a nice place with different beverage options. I would recommend to drink some alcohol ",
                            "This place is a really nice place with table service. I think that the waiters are generally normal ",
                            "This place is an unique place with classy ambience. I would recommend to go there in special days   ",
                            "This place is really good for families with kids. They have a playground for kids, so you can relax "]

    def append_final_reviews(self, review):
        review_file = open(self.learning_data_root+"generated_reviews.txt", "a")
        review_file.write(review + "\n")

    def generate_text(self, predictions, threshold, length_of_sequence):
        complete_review = ""
        for i in range(predictions.shape[0]):
            if predictions[i] > threshold:
                predicted_text = self.generate_text_intern(self.predictionTextPieces[i], length_of_sequence)
                complete_review = complete_review + predicted_text
        self.append_final_reviews(complete_review)
        return complete_review
        # Adar maybe a safe function to append the complete review into a text file

    def generate_text_intern(self, sentence, length_of_sequence):
        # Turns the sentence into integer for the model
        pattern = [self.char_to_int[char] for char in sentence]
        for i in range(length_of_sequence):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.n_vocab)
            prediction = self.model.predict(x, verbose=1)
            index = np.argmax(prediction)
            result = self.int_to_char[index]
            seq_in = [self.int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        # Turns the prediction into readable text
        prediction_list = [self.int_to_char[value] for value in pattern]
        predicted_text = ''.join(map(str, prediction_list))
        return predicted_text

