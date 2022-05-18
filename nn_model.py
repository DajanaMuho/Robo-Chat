import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from pre_processing_model import clean_text


def bag_of_words(text, vocab):
    tokens = clean_text(text)
    bow = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bow[idx] = 1
    return np.array(bow)


class Model:
    def __init__(self, train_X, train_y):
        self.train_X = train_X
        self.train_y = train_y
        self.input_shape = (len(train_X[0]),)
        self.output_shape = len(train_y[0])
        self.epochs = 200

    def fit(self):
        model = Sequential()
        model.add(Dense(128, input_shape=self.input_shape, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(self.output_shape, activation="softmax"))
        adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=["accuracy"])
        print(model.summary())
        model.fit(x=self.train_X, y=self.train_y, epochs=self.epochs, verbose=1)
        return model

    def predict(self, nn_model,  text, vocab, labels):
        bow = bag_of_words(text, vocab)
        result = nn_model.predict(np.array([bow]))[0]
        thresh = 0.2
        y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]

        y_pred.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in y_pred:
            return_list.append(labels[r[0]])
        return return_list

