import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Permute
from pre_processing_model import clean_text
import matplotlib.pyplot as plt

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
        self.epochs = 10

    def fit(self):
        model = Sequential()
        model.add(Dense(512, input_shape=(len(self.train_X[0]),), activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(len(self.train_y[0]), activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=["accuracy"])
        print(model.summary())
        history = model.fit(x=self.train_X, y=self.train_y, epochs=self.epochs, verbose=1)
        self.visualize_Loss_Accuracy(history.history)
        return model

    def predict(self, nn_model,  text, vocab, labels):
        bow = bag_of_words(text, vocab)
        result = nn_model.predict(np.array([bow]))[0]
        thresh = 0.2
        y_pred = [[idx, res] for idx, res in enumerate(result)] # if res > thresh]

        # needed for visualization
        all_tags_predictions = []
        all_prediction = [[idx, res] for idx, res in enumerate(result)]
        for i in all_prediction:
            all_tags_predictions.append([labels[i[0]], i[1]])  # [tag, prediction value]
        y_pred.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in y_pred:
            return_list.append(labels[r[0]])
        return return_list, all_tags_predictions

    def visualize_Loss_Accuracy(self, history):
        loss = history['loss']
        accuracy = history['accuracy']
        epochs = range(1, self.epochs + 1)
        plt.plot(epochs, loss, 'g', label='Loss')
        plt.plot(epochs, accuracy, 'b', label='Accuracy')
        plt.title('Training Accuracy And Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

