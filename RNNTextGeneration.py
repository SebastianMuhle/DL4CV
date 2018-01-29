from tensorflow.python.keras.models import Sequential
import numpy as np
import sys

class RNNTextGeneration:

    # Same as in the RNN function, has to be changed afterwards
    filename = "NameOfOurFile"
    raw_text = open(filename).read()
    chars = sorted(list(set(raw_text)))

    # Info to check if we loaded the right file
    n_chars = len(raw_text)
    n_vocab = len(chars)
    print("Total Characters: ", n_chars)
    print("Total Vocab: ", n_vocab)

    int_to_char = dict((i, c) for i, c in enumerate(chars))
    char_to_int = dict((c, i) for i, c in enumerate(chars))

    # load the network weights
    filename = "weights-improvement-19-1.9435.hdf5"
    model = Sequential()
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    def generateText(self, threshold, lengthOfSequence):
        completeReview = "Finish"

    def generateTextintern(self, sentence, lengthOfSequence):
        # Turns the sentence into integer for the model
        pattern = [self.char_to_int[char] for char in sentence]
        for i in range(lengthOfSequence):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.n_vocab)
            prediction = self.model.predict(x, verbose=0)
            index = np.argmax(prediction)
            result = self.int_to_char[index]
            seq_in = [self.int_to_char[value] for value in pattern]
            sys.stdout.write(result)
            pattern.append(index)
            pattern = pattern[1:len(pattern)]
        # Turns the prediction into readable text
        predictedText = [self.int_to_char[value] for value in pattern]
        return predictedText

